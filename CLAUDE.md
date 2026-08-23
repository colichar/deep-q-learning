# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A from-scratch reimplementation of DeepMind's DQN Atari papers (Mnih et al. 2013, Mnih et al. 2015), applied to
`ALE/SpaceInvaders-v5`. The goal is to first replicate the papers' setup faithfully, then optimize for local
hardware.

The implementation lives at the project root (`src/`, `scripts/`, `tests/`) and is PyTorch-based. An earlier
TensorFlow implementation (driven from `space_invaders_dqn.ipynb`) has been removed; don't resurrect its patterns.

## Comments

Write comments only when the code cannot explain itself: a non-obvious constraint, a subtle
invariant, a specific bug being worked around, or a design decision whose reasoning isn't visible
from the code alone. Never write a comment that restates what the line already says (e.g. `#
prints hello world` above `print("hello world")`) — if removing the comment wouldn't confuse a
future reader, don't write it.

Keep it terse — one line, why-only. Don't explain the whole mechanism in the comment; trust the
reader to follow the code itself.


## Commands

Dependencies are managed with [uv](https://docs.astral.sh/uv/), not pip.

```
uv sync --extra cpu                        # CPU-only machine
uv sync --extra cuda                       # machine with an NVIDIA GPU
uv run AutoROM --accept-license -y         # fetch the Space Invaders ROM (once per fresh .venv)
uv run pytest                              # run all tests
uv run pytest tests/unit           # unit tests only (fast, no ALE env)
uv run pytest -m "not integration"         # same, by marker
uv run pytest tests/integration    # integration tests only (real ALE env + short training run)
uv run pytest tests/unit/test_replay_memory.py::test_episode_boundaries   # single test
uv run python scripts/train.py     # start a training run; --help for the full flag list
```

- `torch`/`torchvision` are optional extras (`cpu` vs `cuda`, mutually exclusive), not base dependencies — a bare
  `uv sync` with no `--extra` **uninstalls them**. Always pass `--extra`.
- Verify a CUDA install with `uv run python -c "import torch; print(torch.cuda.is_available())"`.
- `uv run <cmd>` syncs `--inexact` implicitly, so it won't remove an already-installed extra, but it also won't
  install torch the first time — run an explicit `uv sync --extra ...` at least once per machine.
- No linter/formatter is configured in this repo.

## Architecture (`src/`)

Training is orchestrated by `agent/agent.py:SpaceInvaderAgent`, which owns the env, both networks, memory, and the
epsilon-greedy schedule, and wires them together each frame:

```
gym.make("ALE/SpaceInvaders-v5", frameskip=1)
        │  (ALE's own frameskip is disabled — the agent does it manually, see below)
        ▼
Preprocessor.step_with_skip   — repeats an action for 4 real ALE frames, max-pools the
                                 last two (flicker removal), grayscale/crop/resize to 84x84
        ▼
ReplayMemory.add_frame        — stores the single new 84x84 uint8 frame (not a stacked state)
        │
        ▼ (every update_main_freq frames, once memory_warmup is past)
ReplayMemory.get_batch        — samples valid indices, reconstructs 4-frame stacked
                                 states on the fly from the ring buffer
        ▼
CNNModelPY (Main + Target)    — DeepMind's conv/FC architecture; Target networks
                                 synced from Main every update_target_freq frames
        ▼
custom_huber_loss + Adam
```

Key design points worth knowing before touching this code:

- **`ReplayMemory` (`utils/replay_memory.py`) stores one frame per slot, not stacked 4-frame states.** States are
  reconstructed by indexing `state_length` consecutive frames out of the circular buffer at sample time
  (`_get_state`). This is what makes a paper-scale (10^6-frame) buffer fit in RAM instead of requiring 4x the
  memory. For a sampled index `i`, `actions[i]`/`rewards[i]`/`terminal[i]` describe the transition into frame `i`:
  `curr_state = _get_state(i - 1)` (state ending one frame before `i`) transitions via `actions[i]` to
  `next_state = _get_state(i)` (state ending at `i`) — get this backwards and the states silently shift by one
  frame relative to the action/reward that produced them. `_valid_index` is the load-bearing piece of correctness
  here: it
  rejects indices without `state_length` frames of real history behind them, and indices whose `state_length`
  preceding frames (not including `i` itself, which is allowed to be the terminal frame) cross an episode
  boundary (`terminal` flag). Any change to write/sample logic must keep both invariants intact — see
  `tests/unit/test_replay_memory.py` for the boundary/wraparound/alignment tests that guard this.
- **`utils/replay_memory_from_disk.py` (`ReplayMemoryFromDisk`) is an orphaned earlier design** (disk-backed
  `torch.utils.data.Dataset`) — it is not imported by the agent and predates the ring-buffer approach above. Treat
  it as historical, not as a second code path to keep in sync.
- **Frame-skip + flicker removal is done in `Preprocessor`, not by ALE**, because pairing them requires the two
  maxed frames to be adjacent — the env is constructed with `frameskip=1` specifically so `step_with_skip` can own
  this (see `feea0db` in git log for the rationale). Don't reintroduce ALE-internal frameskip alongside this.
- **A lost life is treated as terminal** for replay-memory purposes (`life_lost or not alive` in
  `SpaceInvaderAgent.train`), matching the DeepMind papers, even though the episode/env itself continues until all
  lives are gone. The *live* `curr_state` used for action selection is deliberately **not** reset on life loss —
  `Preprocessor.new_state` always shifts-and-appends, so for up to 3 frames after a respawn the acting stack still
  contains pre-death frames. This is intentional, not an oversight: replay sampling is unaffected (`_valid_index`
  already excludes any window crossing a `terminal` frame), and resetting the acting stack to a repeated single
  frame would erase motion/velocity cues right when they matter most. Matches common practice in other DQN
  implementations (e.g. OpenAI Baselines' `EpisodicLifeEnv` + frame-stack combo behaves the same way).
- **Reward is clipped to {-1, 0, 1}** before being stored, per the papers; `episode_reward` (used for logging/plots)
  tracks the raw, unclipped reward separately.
- Default hyperparameters on `SpaceInvaderAgent.__init__` are set to match the DeepMind 2015 paper (1M-frame memory,
  50k-frame warmup, target sync every 10k frames, lr 2.5e-4, discount 0.99) — when replicating the paper, prefer
  changing call-site kwargs (as the integration tests do, with a scaled-down memory/frame budget) over changing
  these defaults.
- `agent.save(path)` / `agent.load(path)` persist three things independently under `path/`: replay memory
  (`.npz`), model + optimizer state (`torch.save`), and training history (pickle) — this exists so training can
  resume across sessions on time-boxed compute (e.g. Kaggle's 9h session cap, see README "Results of first
  training"). `load_train_history` also restores `start_frame_num` so a resumed run continues the frame count
  instead of restarting it.
- Uses `gymnasium` + `ale-py` 0.12.x (`import gymnasium as gym` + `gym.register_envs(ale_py)`), not the legacy
  `gym` package (`gym==0.26.2`/`ale-py==0.8.1` has no Python 3.12 wheels).

## Tests (`tests/`)

Split into `unit/` (no ALE env or ROM, fast — a few do build a gymnasium vector env over scripted, ROM-free
sub-envs) and `integration/` (real ALE env + a small real training/save/load run,
marked with the `integration` pytest marker). `pyproject.toml` puts the repo root on `pythonpath`, so tests import
via `from src....` without a `conftest.py`.
