"""Train the `shakespeare` preset on TinyShakespeare so the engine serves a real model.

    python scripts/train_char.py --steps 1200      # ~30 min on 4 CPU cores

Writes results/shakespeare.pt (weights + vocab). Pacer's serving results do not depend
on weight values; this exists so the engine can be shown generating real text, and so
the paged/chunked engine can be checked against a dense forward on a trained model.
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from pacer.model import PRESETS, Transformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    torch.manual_seed(0)

    text = open("data/tinyshakespeare.txt").read()
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    data = torch.tensor([stoi[c] for c in text])
    n = int(0.95 * len(data))
    train, val = data[:n], data[n:]

    model = Transformer(PRESETS["shakespeare"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def batch(split):
        d = train if split == "train" else val
        ix = torch.randint(len(d) - args.ctx - 1, (args.batch,))
        x = torch.stack([d[i:i + args.ctx] for i in ix])
        y = torch.stack([d[i + 1:i + args.ctx + 1] for i in ix])
        return x, y

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = [F.cross_entropy(model.forward_train(x).flatten(0, 1), y.flatten()).item()
                  for x, y in (batch("val") for _ in range(10))]
        model.train()
        return sum(losses) / len(losses)

    t0 = time.time()
    for step in range(args.steps + 1):
        lr = args.lr * min(1.0, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch("train")
        loss = F.cross_entropy(model.forward_train(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step:5d}  train {loss.item():.3f}  val {evaluate():.3f}  {time.time() - t0:.0f}s", flush=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "val_loss": evaluate()}, "results/shakespeare.pt")


if __name__ == "__main__":
    main()
