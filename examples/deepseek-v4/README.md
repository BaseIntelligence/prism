# DeepSeek-V4 Example

DeepSeek-V4 (`dsv4`) is a reference PRISM submission: a small, weight-tied decoder language model in the
DeepSeek family (RMSNorm pre-norm blocks, multi-head causal self-attention, SwiGLU gated MLP) plus its
training and inference hooks. It is a `kind: full` project, so it can register an architecture family and
a training variant from one submission.

The point of this example is to be honest about scoring. The five optional hooks are part of the contract
surface and are recorded as telemetry, but they do not raise the score on their own. The one lever that
moves the architecture score is genuine learning: the model must drive its loss down during evaluation.
dsv4 is built around that fact.

## Layout

```text
examples/deepseek-v4/
  prism.yaml
  src/
    layers.py        # reusable nn blocks + weight-tied DecoderLM
    model.py         # architecture entrypoint; the single module the container loads
    train.py         # the one shared recipe + the five model-agnostic hooks
  submission_singlefile.py   # copy-paste single-file twin (same model, recipe, hooks)
```

The container loads `src/model.py` as one module and resolves the whole contract off it: it calls
`build_model` and `get_recipe` on that module and resolves the five hooks by attribute lookup on it. That
is why `model.py` does `from train import recipe` and re-exports the hooks, while `train.py` holds the one
shared recipe and the model-agnostic hooks. There is exactly one model and one recipe.

## Run The Local Smoke Check

The smoke check validates wiring and learning direction. It is never score eligible
(`validation.score_eligible=false`); a pass means the model is wired correctly and its loss is dropping,
not that it would earn an official score.

```bash
/droid/prism-challenge/.venv/bin/python -m pytest tests/test_example_deepseek_v4.py -q
```

It drives `run_local_cpu_smoke` with a tiny context (`vocab_size=256`, `sequence_length=16`,
`max_parameters=20_000_000`) and asserts the contract is present, the code is sandbox-clean, and the
final loss is below the initial loss.

## Submit

Submit as a multi-file ZIP with `prism.yaml` at the root (preferred), or as `submission_singlefile.py`
through the single-`.py` route. Sign the payload with your miner hotkey and post it when public
submissions are enabled:

```http
POST /v1/submissions
Content-Type: application/json
```

Keep the recipe learning rate inside the q_recipe window and make sure the model actually reduces its
loss before submitting.

## Learn More

- Full walkthrough: [docs/examples/deepseek-v4.md](../../docs/examples/deepseek-v4.md)
- Submission format: [docs/submissions.md](../../docs/submissions.md)
- Miner flow and checklist: [docs/miner/README.md](../../docs/miner/README.md)
