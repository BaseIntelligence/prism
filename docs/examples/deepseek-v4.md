# DeepSeek-V4 Worked Example

This is a complete, end-to-end walkthrough of the DeepSeek-V4 (`dsv4`) reference submission. It is a
`kind: full` PRISM project: one decoder-LM architecture plus its training and inference hooks. Read it
top to bottom if you are building your first multi-file submission, or jump to a section if you already
know the contract and just want the dsv4 specifics.

The goal of this example is to be honest about what actually moves your score. The five optional hooks
are part of the contract surface, but they are telemetry for scoring. The one lever that changes the
architecture score is genuine learning: the model has to drive its loss down during evaluation. dsv4 is
built around that single fact.

## What DeepSeek-V4 Is

dsv4 is a small, weight-tied decoder language model in the DeepSeek family: RMSNorm pre-norm residual
blocks, multi-head causal self-attention, and a SwiGLU gated MLP (with an optional light mixture-of-
experts path that ships off by default). It is deliberately tiny. At the smoke vocabulary it is about
9.5M parameters, far under the 20M smoke cap and the 150M hard cap. Small is on purpose: fewer
parameters raise the efficiency term, and a compact, well-conditioned model is the most reliable way to
show a clean loss drop in a one-to-two step run.

What makes dsv4 a useful reference is not a clever layer. It is the wiring: a coherent multi-file layout
where a single loaded module exposes the full contract, one model, one recipe, and a learning setup that
actually descends.

## Project Layout

```text
examples/deepseek-v4/
  prism.yaml
  src/
    layers.py
    model.py
    train.py
  submission_singlefile.py
```

Three source files do the work, plus the manifest:

- `src/layers.py` holds the reusable building blocks and the weight-tied `DecoderLM`.
- `src/model.py` is the architecture entrypoint and the single module the evaluator loads.
- `src/train.py` holds the one shared recipe and the five model-agnostic hooks.

`submission_singlefile.py` is a copy-paste twin that inlines the same model, recipe, and hooks into one
file for the single-`.py` submission route. The multi-file project under `src/` is the canonical source
of truth; the single-file artifact mirrors it. This walkthrough quotes the multi-file sources only.

## The prism.yaml Manifest

The manifest declares the project kind and the architecture and training entrypoints. dsv4 ships exactly
this:

```yaml
kind: full
architecture:
  entrypoint: src/model.py
  files:
    - src/layers.py
training:
  entrypoint: src/train.py
  files:
    - src/layers.py
```

`kind: full` means this submission can create or update an architecture family and register a training
variant in one go. `architecture.entrypoint` points at `src/model.py`; `training.entrypoint` points at
`src/train.py`. The `files` lists declare the sibling modules each side pulls in. This matches the
"Example Multi-File Project" manifest in [docs/submissions.md](../submissions.md), which is the layout
dsv4 follows.

## How The Container Loads The Project (The Key Idiom)

This is the single most important thing to understand about a `kind: full` project, and it is why the
files are split the way they are.

The container loads the architecture entrypoint, `src/model.py`, as one module. It then resolves the
whole contract off that one module: it calls `build_model(ctx)` and `get_recipe(ctx)` on it, and it
resolves each of the five hooks by attribute lookup on that same module. The training entrypoint is not
loaded as a second module and queried separately. Anything the evaluator needs has to be reachable from
`model.py`.

That single fact drives the design:

- `src/model.py` defines the model and `build_model`, delegates `get_recipe` to the shared recipe, and
  re-exports the five hooks so attribute lookup finds them on `model.py`.
- `src/train.py` owns the shared recipe and the five hooks. Every hook is model-agnostic: it operates on
  the `model` argument the container passes in, never on a model defined inside `train.py`. There is no
  second model and no second recipe.

Because `model.py` imports from `train.py` (and both import from `layers.py`), there is exactly one model
definition and exactly one recipe for the project. `train.py` never imports `model.py`, so there is no
import cycle. The declared sibling files are on the import path at load time, so `from train import ...`
and `from layers import ...` resolve at runtime and pass the sandbox import check.

## File Roles

### src/layers.py: The Reusable Blocks

`layers.py` is plain, sandbox-clean PyTorch: `RMSNorm`, `CausalSelfAttention`, `GatedMLP`, an optional
`LightMoE`, the `TransformerBlock` that wires them pre-norm, and the `DecoderLM` that stacks the blocks
behind a weight-tied head. The weight tie is the detail that matters for learning. The output projection
shares the embedding tensor:

```python
        self.norm_final = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
```

The forward pass is a standard decoder stack returning logits shaped `[B, T, vocab]`:

```python
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
        return self.lm_head(x)
```

`layers.py` is the dependency both `model.py` and `train.py` lean on. It does not need to know anything
about the evaluator beyond the context type.

### src/model.py: The Architecture Entrypoint

`model.py` is the one module the container loads, so it has to present the full contract. It does three
things: it composes the architecture, it delegates the recipe, and it re-exports the hooks.

It composes rather than reinvents. `DeepSeekV4` subclasses the weight-tied `DecoderLM` from `layers.py`
and adds a controlled initialization so the loss is genuinely reducible within one or two steps:

```python
    def _init_for_fast_learning(self) -> None:
        # In-place re-init preserves the lm_head <-> token_emb weight tie (same
        # tensor object set in DecoderLM.__init__). A controlled std yields a
        # high-but-reducible initial loss.
        with torch.no_grad():
            self.token_emb.weight.normal_(0.0, EMB_INIT_STD)
```

The sizing is small literals, with the embedding init std chosen to keep the initial logits non-uniform
but well-conditioned:

```python
MODEL_DIM = 384
MODEL_HEADS = 4
MODEL_LAYERS = 4
MODEL_MLP_RATIO = 4
MODEL_USE_MOE = False
MODEL_NUM_EXPERTS = 4
MODEL_TOP_K = 2

# Embedding (== weight-tied output head) init std. A moderate value keeps the
# initial logits non-uniform (initial_loss meaningfully above ln(vocab)) yet
# well-conditioned, so a single clipped-AdamW step reliably reduces the loss.
EMB_INIT_STD = 0.13
```

It delegates the recipe. `get_recipe` does not define its own recipe; it returns the one shared recipe
from `train.py`, so the whole project has a single recipe definition:

```python
def get_recipe(ctx: PrismContext) -> TrainingRecipe:
    # Delegate to the single shared recipe defined in train.py (canonical idiom:
    # tests/test_training_competitions.py:22-39, docs/submissions.md:205-225).
    return recipe(ctx)
```

It re-exports the hooks. The five hooks live in `train.py`; `model.py` imports them so attribute lookup
on the loaded module finds them:

```python
from train import (
    compute_loss,
    configure_optimizer,
    infer,
    inference_logits,
    recipe,
    train_step,
)
```

With these imports plus `build_model` and `get_recipe` defined locally, the single loaded module exposes
`build_model`, `get_recipe`, and all five hooks, all backed by one model and one recipe.

### src/train.py: The Shared Recipe And Hooks

`train.py` is not loaded as the entrypoint. It is the module `model.py` imports for the recipe and the
hooks. It defines no model class, no `build_model`, and no `get_recipe`; every hook works on the `model`
argument the container hands in.

It declares two distinct learning rates, and the difference between them is the heart of how dsv4 scores.
The recipe rate is a conservative descriptor; the effective rate is what actually trains the model:

```python
RECIPE_LEARNING_RATE = 0.001
RECIPE_BATCH_SIZE = 4
RECIPE_WEIGHT_DECAY = 0.01
```

```python
EFFECTIVE_LEARNING_RATE = 0.005
GRAD_CLIP_NORM = 1.0
```

The shared recipe uses the recipe rate, kept inside the q_recipe window so recipe quality scores 1.0:

```python
def recipe(ctx: PrismContext) -> TrainingRecipe:
    # Shared training recipe. model.py's get_recipe delegates here so there is
    # exactly ONE recipe definition for the whole project.
    return TrainingRecipe(
        learning_rate=RECIPE_LEARNING_RATE,
        batch_size=RECIPE_BATCH_SIZE,
        optimizer="adamw",
        scheduler="cosine",
        weight_decay=RECIPE_WEIGHT_DECAY,
    )
```

## The Five Hooks

dsv4 defines all five optional hooks for contract completeness. They are recorded as telemetry by the
evaluator. Defining them does not raise your score; what they enable (a real optimizer and a real loss)
is what lets the model learn, and learning is the lever. Keep that distinction clear: presence is
contract, descent is score.

### configure_optimizer

This hook builds the optimizer that performs the actual gradient steps. It uses the effective rate, not
the recipe rate, so the model descends within one or two updates. Supplying a custom AdamW here also
bypasses the evaluator fallback cap that would otherwise clamp the rate:

```python
def configure_optimizer(model, recipe, ctx):
    """Build the training optimizer for the supplied model.

    The harness reads two distinct learning rates. The recipe's declared
    learning_rate is a static descriptor that the scorer keeps inside the
    q_recipe window [1e-5, 3e-3]; it is intentionally conservative. The
    effective AdamW rate used for the actual gradient steps is set here so the
    model descends within one or two updates. Supplying our own AdamW bypasses
    the container's min(recipe.lr, 3e-4) fallback cap (container.py:852-862).

    Keeping the effective rate in this one place (EFFECTIVE_LEARNING_RATE) means
    an ablated twin only has to change this single value to disable learning
    while leaving every other hook identical.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable,
        lr=EFFECTIVE_LEARNING_RATE,
        weight_decay=RECIPE_WEIGHT_DECAY,
    )
```

### inference_logits And infer (Precedence)

dsv4 ships both inference paths, and the order matters. When both exist, the container resolves
`inference_logits` first; `infer` is present for completeness but stays unused by precedence. Document
this in your own submission so you know which path runs.

`inference_logits` is the preferred path and returns raw logits `[B, T, V]`:

```python
def inference_logits(model, batch, ctx):
    """Return raw logits [B, T, V]; this is the preferred inference path.

    The container resolves inference_logits before infer (container.py:833-840).
    """
    return model(batch.tokens)
```

`infer` returns greedy next-token predictions and stays callable, but precedence means it is not the
path used when `inference_logits` is present:

```python
def infer(model, batch, ctx):
    """Return greedy next-token predictions.

    Present for contract completeness. The container resolves inference_logits
    before infer, so this path stays callable but unused by precedence.
    """
    logits = model(batch.tokens)
    return logits.argmax(dim=-1)
```

### compute_loss

Next-token cross-entropy over the forward output. The harness supplies tokens and targets already
aligned as a next-token pair; when targets are absent the shift is reconstructed locally, so this never
returns `None`:

```python
def compute_loss(model, batch, ctx):
    """Next-token cross-entropy over forward output [B, T, V].

    The harness supplies tokens and targets already aligned as a next-token
    pair (targets = tokens shifted by one, container.py:829-830). When targets
    are absent the shift is reconstructed locally. Always returns a torch.Tensor.
    """
    logits = model(batch.tokens)
    vocab = logits.shape[-1]
    targets = batch.targets
    if targets is None:
        logits = logits[:, :-1, :]
        targets = batch.tokens[:, 1:]
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1) % vocab
    return F.cross_entropy(flat_logits, flat_targets)
```

### train_step

One optimization step that returns the pre-step loss tensor, with gradient clipping:

```python
def train_step(model, batch, optimizer, ctx):
    """Run one optimization step and return the pre-step loss tensor."""
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss(model, batch, ctx)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    return loss
```

## How Learning Is Achieved

This is the section that explains the score. The architecture quality signal is improvement divided by
initial loss: the model has to start at a meaningful loss and reduce it. Two design choices make that
happen reliably, and they pull in slightly different directions on purpose.

First, the two learning rates serve two different readers. The recipe's declared `learning_rate` is a
static descriptor. The scorer keeps it inside the q_recipe window `[1e-5, 3e-3]`, so dsv4 sets it to
`0.001` and earns full recipe quality. That rate is intentionally conservative and is not the rate that
trains the model. The rate that trains the model is the effective AdamW rate set inside
`configure_optimizer`, which is `0.005`. By building its own AdamW, dsv4 bypasses the evaluator fallback
that would otherwise cap the rate at `min(recipe.lr, 3e-4)`. The effective rate sits above that cap, so
the gradient steps are large enough to descend, while the declared recipe rate stays inside the window
that scores recipe quality at 1.0.

Second, the model is small and the embedding is initialized for a high-but-reducible start. The
controlled init std keeps the initial logits non-uniform, so the initial loss is meaningfully above the
uniform baseline, which leaves real room to improve. The weight-tied head means a single clipped AdamW
step moves both the embedding and the output projection together. Keeping the parameter count small
(about 9.5M at the smoke vocabulary) also raises the efficiency term, `1/(1+log10(params))`, so a compact
model that still learns scores better than a large one that learns the same amount.

Put together: a high-but-reducible initial loss, an effective rate large enough to descend, clipped
steps for stability, and a small parameter count for efficiency. During the smoke run the loss falls
across the steps (for example, from roughly 12.6 to roughly 6.9 over two steps), which is exactly the
direction the scorer reads as quality. The hooks make this possible; the descent is what scores.

## Running The Local Smoke Check

The local CPU smoke check validates wiring and learning direction. It is never score eligible: the
manifest sets `validation.score_eligible=false`, so it proves the model is wired correctly and moving in
the right direction, not that it would earn an official score. Run it with:

```bash
/droid/prism-challenge/.venv/bin/python -m pytest tests/test_example_deepseek_v4.py -q
```

Under the hood this drives `run_local_cpu_smoke` with a tiny context (`vocab_size=256`,
`sequence_length=16`, `max_parameters=20_000_000`), reads the run manifest, and asserts the contract is
present, the submission is sandbox-clean, and the final loss is below the initial loss. The last
assertion is the one that matters: it confirms the model genuinely learns during the run. The smoke run
sets `validation.score_eligible` to false, so a pass here is a wiring-and-learning check, not a score.

For the broader wiring check across smoke modes you can also run the shared local smoke from the miner
guide:

```bash
pytest tests/test_local_cpu_smoke_eval.py -q
```

## Submitting

dsv4 is a `kind: full` project, so it can be submitted as a multi-file ZIP with `prism.yaml` at the root,
or as the single-file `submission_singlefile.py` through the single-`.py` route. The multi-file layout is
preferred because the manifest lets the evaluator attribute architecture and training files separately.

Submit through the public submission route when public submissions are enabled, signing the payload with
your miner hotkey:

```http
POST /v1/submissions
Content-Type: application/json
```

```json
{
  "hotkey": "5Abc...",
  "signature": "<sr25519-signature>",
  "timestamp": 1760000000,
  "nonce": "unique-nonce",
  "code": "<base64-or-text-submission-payload>"
}
```

Before you submit: run the smoke check, confirm the manifest and entrypoints exist, keep the recipe rate
inside the q_recipe window, and make sure the model actually reduces its loss. See
[docs/submissions.md](../submissions.md) for the full submission format and
[docs/miner/README.md](../miner/README.md) for the miner flow and checklist.
