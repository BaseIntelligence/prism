"""Staging e2e miner A training: cosine LR + label smoothing on the pinned shard."""

import math
import time

import pyarrow.parquet as pq
import torch
from transformers import GPT2TokenizerFast

try:
    import prism_telemetry
except ImportError:
    class _TelemetryFallback:
        @staticmethod
        def report(**_kwargs):
            return None

        @staticmethod
        def finish_evaluation():
            return None

    prism_telemetry = _TelemetryFallback()


def _texts(path, n):
    table = pq.read_table(path, columns=["text"])
    xs = [t for t in table.column("text").to_pylist() if isinstance(t, str) and len(t) >= 200]
    return xs[:n]


def train(model, ctx):
    """Recipe contract entrypoint: returns a metrics dict (val is harness-side)."""
    device = ctx["device"]
    torch.manual_seed(int(ctx["seed"]))
    guard = ctx.get("guard", lambda: None)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    texts = _texts(ctx["dataset_path"], int(ctx.get("train_rows", 2048)))
    block = model.block if hasattr(model, "block") else 512

    g = torch.Generator().manual_seed(int(ctx["seed"]))
    perm = torch.randperm(len(texts), generator=g).tolist()

    max_steps = int(ctx.get("max_train_steps", 20000))
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.05)
    model.train()
    steps = 0
    bs = 8
    last = 0.0
    t0 = time.time()
    for i in range(0, min(2000, len(perm) - bs), bs):
        guard()
        lr_scale = 0.5 * (1.0 + math.cos(math.pi * min(1.0, steps / max(1, 500))))
        for group in opt.param_groups:
            group["lr"] = 6e-4 * lr_scale
        batch_txt = [texts[j] for j in perm[i : i + bs]]
        enc = tok(
            batch_txt, return_tensors="pt", truncation=True, max_length=block, padding=True
        ).to(device)
        ids = enc.input_ids
        out = model(ids[:, :-1])
        logits = out.logits if hasattr(out, "logits") else out
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
            ignore_index=tok.pad_token_id,
            label_smoothing=0.05,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        last = float(loss.item())
        steps += 1
        if steps == 1 or steps % 10 == 0:
            prism_telemetry.report(
                loss=last,
                step=steps,
                grad_norm=grad_norm,
                layer_stats={"mixer_head": {"grad_norm": grad_norm}},
            )
        if steps >= max_steps:
            break
        if steps >= 300 and last < 4.0:
            # Early stop: score the model as-is before the cap.
            break
    prism_telemetry.finish_evaluation()
    return {"train_loss": last, "train_steps": steps, "train_seconds": time.time() - t0}

# sim-bpb-tuning: a-28
