# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A from-scratch reimplementation of DeepMind's DQN Atari papers (Mnih et al. 2013, Mnih et al. 2015), applied to
`ALE/SpaceInvaders-v5`. The goal is to first replicate the papers' setup faithfully, then optimize for local
hardware.

**`pytorch/` is the active implementation — work here unless told otherwise.** The top-level `src/` directory
(TensorFlow + legacy `gym`, driven from `space_invaders_dqn.ipynb`) is deprecated and slated for removal; don't
extend it, and don't copy patterns from it into `pytorch/`.

## Commands

Dependencies are managed with [uv](https://docs.astral.sh/uv/), not pip.

```
uv sync --extra cpu                        # CPU-only machine
uv sync --extra cuda                       # machine with an NVIDIA GPU
uv run AutoROM --accept-license -y         # fetch the Space Invaders ROM (once per fresh .venv)
uv run pytest                              # run all tests
uv run pytest pytorch/tests/unit           # unit tests only (fast, no ALE env)
uv run pytest -m "not integration"         # same, by marker
uv run pytest pytorch/tests/integration    # integration tests only (real ALE env + short training run)
uv run pytest pytorch/tests/unit/test_replay_memory.py::test_episode_boundaries   # single test
uv run python pytorch/scripts/train.py     # start a training run; --help for the full flag list
```

- `torch`/`torchvision` are optional extras (`cpu` vs `cuda`, mutually exclusive), not base dependencies — a bare
  `uv sync` with no `--extra` **uninstalls them**. Always pass `--extra`.
- Verify a CUDA install with `uv run python -c "import torch; print(torch.cuda.is_available())"`.
- `uv run <cmd>` syncs `--inexact` implicitly, so it won't remove an already-installed extra, but it also won't
  install torch the first time — run an explicit `uv sync --extra ...` at least once per machine.
- No linter/formatter is configured in this repo.

## Architecture (`pytorch/src/`)

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
  memory. `_valid_index` is the load-bearing piece of correctness here: it rejects indices too close to the write
  head (stale `next_state`) and indices whose 4-frame window would cross an episode boundary (`terminal` flag).
  Any change to write/sample logic must keep both invariants intact — see
  `pytorch/tests/unit/test_replay_memory.py` for the boundary/wraparound tests that guard this.
- **`utils/replay_memory_from_disk.py` (`ReplayMemoryFromDisk`) is an orphaned earlier design** (disk-backed
  `torch.utils.data.Dataset`) — it is not imported by the agent and predates the ring-buffer approach above. Treat
  it as historical, not as a second code path to keep in sync.
- **Frame-skip + flicker removal is done in `Preprocessor`, not by ALE**, because pairing them requires the two
  maxed frames to be adjacent — the env is constructed with `frameskip=1` specifically so `step_with_skip` can own
  this (see `feea0db` in git log for the rationale). Don't reintroduce ALE-internal frameskip alongside this.
- **A lost life is treated as terminal** for replay-memory purposes (`life_lost or not alive` in
  `SpaceInvaderAgent.train`), matching the DeepMind papers, even though the episode/env itself continues until all
  lives are gone.
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
- `gymnasium` + `ale-py` 0.12.x is what `pytorch/` uses (`import gymnasium as gym` + `gym.register_envs(ale_py)`);
  the legacy `import gym` (`gym==0.26.2`/`ale-py==0.8.1`) is only used by the deprecated `src/` TensorFlow code and
  has no Python 3.12 wheels — don't mix the two APIs.

## Tests (`pytorch/tests/`)

Split into `unit/` (no gym env, fast) and `integration/` (real ALE env + a small real training/save/load run,
marked with the `integration` pytest marker). `pyproject.toml` puts `pytorch/` on `pythonpath`, so tests import via
`from src....` without a `conftest.py`.
