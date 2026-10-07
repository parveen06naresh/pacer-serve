// Crash Replay engine: series decoding, portfolio accounting, the quant
// opponent and the random-trader skill test. No DOM; runs in the browser
// and in Node (for tests).
(function (root) {
  "use strict";

  const START_CASH = 10000;
  const COST = 0.0005; // 5 bps of traded value per trade
  const WARMUP = 260; // trading days of history the quant needs

  const DAY = 86400000;

  // --- series -------------------------------------------------------------
  function decode(raw) {
    const n = raw.close.length;
    const t = new Float64Array(n);
    t[0] = Date.parse(raw.start + "T00:00:00Z");
    for (let i = 1; i < n; i++) t[i] = t[i - 1] + (raw.gaps.charCodeAt(i - 1) - 48) * DAY;
    return { t, close: raw.close, rf: raw.rf, dy: raw.dy, n };
  }

  function iso(ms) {
    return new Date(ms).toISOString().slice(0, 10);
  }

  // first index with date >= d
  function indexOf(s, d) {
    const target = Date.parse(d + "T00:00:00Z");
    let lo = 0, hi = s.n - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (s.t[mid] < target) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  // Daily returns for the window [i0, i1]: r[k] is the return earned from
  // close i0+k to close i0+k+1, for the market (price + dividends) and cash.
  function windowReturns(s, i0, i1) {
    const m = i1 - i0;
    const mkt = new Float64Array(m), cash = new Float64Array(m);
    for (let k = 0; k < m; k++) {
      const i = i0 + k;
      const days = (s.t[i + 1] - s.t[i]) / DAY;
      mkt[k] = s.close[i + 1] / s.close[i] - 1 + (s.dy[i] / 100) * days / 365;
      cash[k] = (s.rf[i] / 100) * days / 365;
    }
    return { mkt, cash };
  }

  // --- portfolio ----------------------------------------------------------
  // A book holds a value and an exposure (0..1). Decisions are made at a
  // close and earn the next day's return, so nothing ever sees the future.
  function Book(exposure) {
    this.value = START_CASH;
    this.exposure = exposure;
    this.peak = START_CASH;
    this.maxDD = 0;
    this.trades = 0;
    this.path = [START_CASH];
    this.daily = [];
  }
  Book.prototype.set = function (e) {
    if (Math.abs(e - this.exposure) < 1e-9) return false;
    this.value -= COST * Math.abs(e - this.exposure) * this.value;
    this.exposure = e;
    this.trades++;
    return true;
  };
  Book.prototype.step = function (mkt, cash) {
    const r = this.exposure * mkt + (1 - this.exposure) * cash;
    this.value *= 1 + r;
    this.daily.push(r);
    this.path.push(this.value);
    if (this.value > this.peak) this.peak = this.value;
    const dd = 1 - this.value / this.peak;
    if (dd > this.maxDD) this.maxDD = dd;
  };

  function stats(book, cash) {
    const n = book.daily.length;
    let mean = 0, sq = 0, worst = 0, excess = 0;
    for (let k = 0; k < n; k++) {
      const x = book.daily[k] - cash[k];
      excess += x;
      mean += x;
      if (book.daily[k] < worst) worst = book.daily[k];
    }
    mean /= Math.max(n, 1);
    for (let k = 0; k < n; k++) sq += (book.daily[k] - cash[k] - mean) ** 2;
    const sd = Math.sqrt(sq / Math.max(n - 1, 1));
    return {
      final: book.value,
      ret: book.value / START_CASH - 1,
      maxDD: book.maxDD,
      worstDay: worst,
      sharpe: sd > 0 ? (mean / sd) * Math.sqrt(252) : 0,
      trades: book.trades,
    };
  }

  // --- the quant ----------------------------------------------------------
  // Classic two-rule system used by trend-following funds:
  //   1. trend filter: invested only while the close is above its 200-day average
  //   2. volatility target: size = min(1, 15% / realised 20-day volatility)
  // Re-sized at each close; only trades when the size moves by 10 points+.
  function quantSignals(s, i0, i1) {
    const out = new Float64Array(i1 - i0 + 1);
    for (let i = i0; i <= i1; i++) {
      let sum = 0;
      for (let j = i - 199; j <= i; j++) sum += s.close[j];
      const sma = sum / 200;
      let m = 0, v = 0;
      const lr = [];
      for (let j = i - 19; j <= i; j++) lr.push(Math.log(s.close[j] / s.close[j - 1]));
      for (const x of lr) m += x;
      m /= lr.length;
      for (const x of lr) v += (x - m) ** 2;
      const vol = Math.sqrt(v / (lr.length - 1)) * Math.sqrt(252);
      const size = s.close[i] > sma ? Math.min(1, 0.15 / Math.max(vol, 1e-6)) : 0;
      out[i - i0] = Math.round(size * 20) / 20;
    }
    return out;
  }

  function runQuant(signals, ret) {
    const q = new Book(signals[0]);
    const pos = [q.exposure];
    for (let k = 0; k < ret.mkt.length; k++) {
      q.step(ret.mkt[k], ret.cash[k]);
      const want = signals[k + 1];
      if (Math.abs(want - q.exposure) >= 0.1 || (want === 0 && q.exposure !== 0)) q.set(want);
      pos.push(q.exposure);
    }
    return { book: q, pos };
  }

  function runHold(ret) {
    const b = new Book(1);
    for (let k = 0; k < ret.mkt.length; k++) b.step(ret.mkt[k], ret.cash[k]);
    return b;
  }

  // --- skill test -----------------------------------------------------------
  // Random traders who make exactly as many trades as the player, on random
  // days, to random positions. Beating most of them is evidence of skill,
  // not luck (a permutation test).
  function skillTest(ret, trades, final, sims, seed) {
    sims = sims || 2000;
    let x = seed || 12345;
    const rand = () => ((x = (x * 1664525 + 1013904223) >>> 0) / 4294967296);
    const m = ret.mkt.length;
    const levels = [0, 0.5, 1];
    let beaten = 0;
    const finals = new Float64Array(sims);
    for (let s = 0; s < sims; s++) {
      const days = new Set();
      while (days.size < Math.min(trades, m)) days.add(Math.floor(rand() * m));
      let v = START_CASH, e = 1;
      for (let k = 0; k < m; k++) {
        if (days.has(k)) {
          let ne = e;
          while (ne === e) ne = levels[Math.floor(rand() * 3)];
          v -= COST * Math.abs(ne - e) * v;
          e = ne;
        }
        v *= 1 + e * ret.mkt[k] + (1 - e) * ret.cash[k];
      }
      finals[s] = v;
      if (final > v) beaten++;
    }
    return { pct: beaten / sims, finals };
  }

  // --- drawdown of the index itself (for scenario cards) ----------------------
  function indexDrawdown(s, i0, i1) {
    let peak = s.close[i0], peakI = i0, best = 0, from = i0, to = i0;
    for (let i = i0; i <= i1; i++) {
      if (s.close[i] > peak) { peak = s.close[i]; peakI = i; }
      const dd = 1 - s.close[i] / peak;
      if (dd > best) { best = dd; from = peakI; to = i; }
    }
    return { dd: best, peak: from, trough: to };
  }

  // Windows of `len` days that contain a fall of at least `minDD` from a
  // peak inside the window. Used for the mystery crash.
  function mysteryWindows(s, len, minDD, firstYear) {
    const first = indexOf(s, firstYear + "-01-01");
    const out = [];
    for (let i0 = Math.max(first, WARMUP + 1); i0 + len < s.n; i0 += 21) {
      if (indexDrawdown(s, i0, i0 + len).dd >= minDD) out.push(i0);
    }
    return out;
  }

  root.CR = {
    START_CASH, COST, WARMUP, decode, iso, indexOf, windowReturns, Book,
    stats, quantSignals, runQuant, runHold, skillTest, indexDrawdown, mysteryWindows,
  };
})(typeof window !== "undefined" ? window : globalThis);
