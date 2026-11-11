// pacer._core: C++17 hot path of the Pacer scheduler, exposed to Python with pybind11.
//
// Each scheduling step Pacer (1) orders waiting prefills with Moore-Hodgson,
// (2) binary-searches the largest prefill token budget whose predicted step latency
// fits the TPOT target, and (3) re-calibrates its risk margin with adaptive conformal
// inference. The Python versions allocate StepShape tuples and numpy arrays inside
// every probe of the search; here each probe is an O(k) pass over flat arrays, with
// no allocation. Results are bit-for-bit the same decisions (tests/test_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------- KV block allocator
// Same semantics (and same block ids, in the same order) as pacer.engine.BlockManager.
class BlockAllocator {
 public:
  BlockAllocator(int64_t num_blocks, int64_t block_size) : num_blocks_(num_blocks), block_size_(block_size) {
    if (num_blocks <= 0 || block_size <= 0) throw std::invalid_argument("num_blocks and block_size must be > 0");
    free_.reserve(num_blocks);
    for (int64_t b = num_blocks - 1; b >= 0; --b) free_.push_back(b);
  }
  int64_t blocks_needed(int64_t tokens) const { return (tokens + block_size_ - 1) / block_size_; }
  bool can_admit(int64_t tokens) const { return static_cast<int64_t>(free_.size()) >= blocks_needed(tokens); }
  std::vector<int64_t> admit(int64_t tokens) {
    const int64_t n = blocks_needed(tokens);
    if (static_cast<int64_t>(free_.size()) < n) throw std::runtime_error("out of KV blocks");
    std::vector<int64_t> out(n);
    for (int64_t i = 0; i < n; ++i) { out[i] = free_.back(); free_.pop_back(); }
    return out;
  }
  void release(const std::vector<int64_t>& blocks) { free_.insert(free_.end(), blocks.rbegin(), blocks.rend()); }
  // Physical cache slot of each position in [start, end) for a request's block table.
  std::vector<int64_t> slots(const std::vector<int64_t>& blocks, int64_t start, int64_t end) const {
    std::vector<int64_t> out;
    out.reserve(std::max<int64_t>(end - start, 0));
    for (int64_t p = start; p < end; ++p) out.push_back(blocks.at(p / block_size_) * block_size_ + p % block_size_);
    return out;
  }
  int64_t num_free() const { return static_cast<int64_t>(free_.size()); }
  double utilization() const { return 1.0 - static_cast<double>(free_.size()) / num_blocks_; }

 private:
  int64_t num_blocks_, block_size_;
  std::vector<int64_t> free_;
};

// ------------------------------------------------------- budget search (linear model)
// Features (pacer.costmodel.FEATURES): bias, tokens, prefill_attn, decode_kv,
// decode_pad, prefill_seqs, decode_seqs.  Prefill candidates are packed greedily in
// order: candidate i gets min(remaining_i, budget_left) tokens.
struct BudgetPlanner {
  std::vector<double> coef;
  explicit BudgetPlanner(std::vector<double> c) : coef(std::move(c)) {
    if (coef.size() != 7) throw std::invalid_argument("linear latency model needs 7 coefficients");
  }

  double predict(const std::vector<int64_t>& remaining, const std::vector<int64_t>& prefilled,
                 double dec_sum, double dec_pad, double n_dec, int64_t budget) const {
    double tokens = n_dec, attn = 0.0, seqs = 0.0;
    int64_t left = budget;
    for (size_t i = 0; i < remaining.size() && left > 0; ++i) {
      const int64_t n = std::min(remaining[i], left);
      tokens += n;
      attn += static_cast<double>(n) * static_cast<double>(prefilled[i] + n);
      seqs += 1.0;
      left -= n;
    }
    return coef[0] + coef[1] * tokens + coef[2] * attn + coef[3] * dec_sum + coef[4] * dec_pad +
           coef[5] * seqs + coef[6] * n_dec;
  }

  // Largest budget in [0, hi] with predicted latency <= target (same search as Python).
  int64_t plan(const std::vector<int64_t>& remaining, const std::vector<int64_t>& prefilled,
               const std::vector<int64_t>& decode_ctx, double target, int64_t hi) const {
    if (remaining.size() != prefilled.size()) throw std::invalid_argument("remaining/prefilled size mismatch");
    if (std::isinf(target)) return hi;
    double dec_sum = 0.0, dec_max = 0.0;
    for (int64_t d : decode_ctx) { dec_sum += d; dec_max = std::max(dec_max, static_cast<double>(d)); }
    const double n_dec = static_cast<double>(decode_ctx.size());
    const double dec_pad = decode_ctx.empty() ? 0.0 : n_dec * dec_max;
    int64_t lo = 0, c = 0;
    while (lo <= hi) {
      const int64_t mid = lo + (hi - lo) / 2;
      if (predict(remaining, prefilled, dec_sum, dec_pad, n_dec, mid) <= target) { c = mid; lo = mid + 1; }
      else hi = mid - 1;
    }
    return c;
  }
};

// ------------------------------------------------------------------ Moore-Hodgson
// Order jobs to minimise the number that miss their deadline (1||sum U_j), O(n log n).
// Returns indices: on-time jobs in arrival order, then late jobs in arrival order.
// Ties match Python's heapq on (-cost, rid): the evicted job is the longest, lowest rid.
std::vector<int64_t> moore_hodgson(const std::vector<double>& arrival, const std::vector<double>& cost,
                                   const std::vector<int64_t>& rid, double now, double ttft) {
  const size_t n = arrival.size();
  if (cost.size() != n || rid.size() != n) throw std::invalid_argument("size mismatch");
  std::vector<int64_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return arrival[a] < arrival[b]; });
  auto cmp = [&](int64_t a, int64_t b) {  // max-heap on cost, then min rid
    if (cost[a] != cost[b]) return cost[a] < cost[b];
    return rid[a] > rid[b];
  };
  std::priority_queue<int64_t, std::vector<int64_t>, decltype(cmp)> heap(cmp);
  std::vector<char> late(n, 0);
  double t = 0.0;
  for (int64_t i : idx) {
    heap.push(i);
    t += cost[i];
    if (now + t > arrival[i] + ttft) {
      const int64_t e = heap.top();
      heap.pop();
      t -= cost[e];
      late[e] = 1;
    }
  }
  std::vector<int64_t> out;
  out.reserve(n);
  for (int64_t i : idx) if (!late[i]) out.push_back(i);
  for (int64_t i : idx) if (late[i]) out.push_back(i);
  return out;
}

// ------------------------------------------------ adaptive conformal inference
// Same update as pacer.costmodel.OnlineConformal (Gibbs & Candes, 2021), with a ring
// buffer and nth_element instead of copying the window into numpy every step.
class OnlineConformal {
 public:
  OnlineConformal(double alpha, double gamma, size_t window, size_t warmup)
      : alpha_(alpha), alpha_t_(alpha), gamma_(gamma), window_(window), warmup_(warmup) {
    if (window == 0) throw std::invalid_argument("window must be > 0");
    buf_.reserve(window);
  }
  bool ready() const { return buf_.size() >= warmup_; }
  double bound() const {
    if (buf_.empty()) throw std::runtime_error("no observations");
    scratch_ = buf_;
    if (alpha_t_ <= 0) return *std::max_element(scratch_.begin(), scratch_.end()) * 1.25;
    if (alpha_t_ >= 1) return *std::min_element(scratch_.begin(), scratch_.end());
    // numpy quantile(method="higher"): element at ceil((n - 1) * q) of the sorted window.
    const double q = 1.0 - alpha_t_;
    const size_t k = static_cast<size_t>(std::ceil((scratch_.size() - 1) * q));
    std::nth_element(scratch_.begin(), scratch_.begin() + k, scratch_.end());
    return scratch_[k];
  }
  double center() const {  // numpy median
    scratch_ = buf_;
    const size_t n = scratch_.size(), m = n / 2;
    std::nth_element(scratch_.begin(), scratch_.begin() + m, scratch_.end());
    const double hi = scratch_[m];
    if (n % 2) return hi;
    const double lo = *std::max_element(scratch_.begin(), scratch_.begin() + m);
    return 0.5 * (lo + hi);
  }
  void update(double ratio) {
    if (ready()) {
      const bool miss = ratio > bound();
      alpha_t_ += gamma_ * (alpha_ - (miss ? 1.0 : 0.0));
      ++n_;
      misses_ += miss;
    }
    if (buf_.size() < window_) buf_.push_back(ratio);
    else { buf_[head_] = ratio; head_ = (head_ + 1) % window_; }
  }
  double alpha_t() const { return alpha_t_; }
  int64_t n() const { return n_; }
  int64_t misses() const { return misses_; }

 private:
  double alpha_, alpha_t_, gamma_;
  size_t window_, warmup_, head_ = 0;
  int64_t n_ = 0, misses_ = 0;
  std::vector<double> buf_;
  mutable std::vector<double> scratch_;
};

PYBIND11_MODULE(_core, m) {
  m.doc() = "Pacer scheduler hot path in C++17";
  py::class_<BlockAllocator>(m, "BlockAllocator")
      .def(py::init<int64_t, int64_t>(), py::arg("num_blocks"), py::arg("block_size"))
      .def("blocks_needed", &BlockAllocator::blocks_needed)
      .def("can_admit", &BlockAllocator::can_admit)
      .def("admit", &BlockAllocator::admit)
      .def("release", &BlockAllocator::release)
      .def("slots", &BlockAllocator::slots)
      .def_property_readonly("num_free", &BlockAllocator::num_free)
      .def_property_readonly("utilization", &BlockAllocator::utilization);
  py::class_<BudgetPlanner>(m, "BudgetPlanner")
      .def(py::init<std::vector<double>>(), py::arg("coef"))
      .def("plan", &BudgetPlanner::plan, py::arg("remaining"), py::arg("prefilled"), py::arg("decode_ctx"),
           py::arg("target"), py::arg("hi"));
  m.def("moore_hodgson", &moore_hodgson, py::arg("arrival"), py::arg("cost"), py::arg("rid"), py::arg("now"),
        py::arg("ttft"));
  py::class_<OnlineConformal>(m, "OnlineConformal")
      .def(py::init<double, double, size_t, size_t>(), py::arg("alpha") = 0.1, py::arg("gamma") = 0.02,
           py::arg("window") = 256, py::arg("warmup") = 32)
      .def_property_readonly("ready", &OnlineConformal::ready)
      .def("bound", &OnlineConformal::bound)
      .def("center", &OnlineConformal::center)
      .def("update", &OnlineConformal::update)
      .def_property_readonly("alpha_t", &OnlineConformal::alpha_t)
      .def_property_readonly("n", &OnlineConformal::n)
      .def_property_readonly("misses", &OnlineConformal::misses);
}
