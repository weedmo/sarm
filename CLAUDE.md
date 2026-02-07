# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is a Hugging Face library for real-world robotics. It provides policies (ML models for robot control), dataset tools, hardware drivers (motors, cameras, robots), simulation environments, and training/evaluation pipelines. The package lives in `src/lerobot/` (src layout).

## Common Commands

### Installation (development)
```bash
pip install -e ".[dev,test]"
# Or with uv (preferred):
uv sync --extra test       # minimal test deps
uv sync --extra all        # all extras (excludes wallx, pi, groot due to conflicts)
```

### Testing
```bash
# Git LFS is required for test artifacts
git lfs install && git lfs pull

pytest tests -vv                              # full suite
pytest -sv tests/test_specific.py             # single file
pytest -sv tests/test_specific.py::test_name  # single test

# End-to-end tests (actually train tiny models)
make test-act-ete-train DEVICE=cuda
make test-end-to-end DEVICE=cpu
```

### Linting and Formatting
```bash
pre-commit run --all-files   # run all checks (ruff, typos, bandit, mypy, etc.)
ruff check .                 # lint only
ruff format .                # format only
ruff check --fix .           # auto-fix
```

### CLI Entry Points
`lerobot-train`, `lerobot-eval`, `lerobot-record`, `lerobot-replay`, `lerobot-calibrate`, `lerobot-teleoperate`, `lerobot-find-cameras`, `lerobot-find-port`, `lerobot-setup-motors`, `lerobot-dataset-viz`, `lerobot-info`, `lerobot-edit-dataset`

## Architecture

### Configuration System
Uses **draccus** (not hydra/argparse). Configs are dataclasses parsed from CLI/JSON/YAML. The `lerobot.configs.parser.wrap()` decorator wraps `draccus.parse()` with path-based loading and plugin discovery. Polymorphic types use `draccus.ChoiceRegistry` (e.g., `--policy.type=act` resolves to `ACTConfig`).

### Policy Structure
Every policy follows a strict naming convention enforced by dynamic imports in `lerobot/policies/factory.py`:
- `configuration_<name>.py` - Config dataclass extending `PreTrainedConfig` (which extends `draccus.ChoiceRegistry`)
- `modeling_<name>.py` - Model class extending `PreTrainedPolicy` (extends `nn.Module`)
- `processor_<name>.py` - Pre/post-processing with factory function `make_<name>_pre_post_processors()`

`PreTrainedPolicy.__init_subclass__` enforces that every subclass defines `config_class` and `name` class attributes at definition time.

Both `PreTrainedConfig` and `PreTrainedPolicy` inherit from a custom `HubMixin` for HF Hub integration (`from_pretrained`, `save_pretrained`, `push_to_hub`).

### Plugin System
Third-party extensions register config classes via `draccus.ChoiceRegistry`. CLI supports `--env.discover_packages_path=my_package` for runtime plugin loading.

### Registry
`src/lerobot/__init__.py` maintains manually-curated lists of `available_policies`, `available_envs`, `available_robots`, `available_cameras`, `available_motors`. This file and `tests/test_available.py` must be updated when adding new components.

### Key Directories
- `src/lerobot/policies/` - Policy implementations (ACT, Diffusion, TDMPC, VQ-BeT, Pi0, Pi0Fast, SmolVLA, SAC, etc.)
- `src/lerobot/datasets/` - LeRobotDataset format (Parquet + MP4), data loading, video utils
- `src/lerobot/robots/` - Robot hardware abstractions
- `src/lerobot/cameras/`, `motors/`, `teleoperators/` - Hardware drivers
- `src/lerobot/configs/` - Dataclass-based config system
- `src/lerobot/envs/` - Simulation environments (Aloha, PushT, LIBERO, MetaWorld)
- `src/lerobot/scripts/` - CLI entry point implementations
- `src/lerobot/utils/` - Shared utilities

## Code Style
- Python 3.10 target
- Line length: 110
- Ruff for linting/formatting (double quotes, space indent)
- isort with `known-first-party = ["lerobot"]`
- mypy partially enabled (strict for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.cameras`, `lerobot.transport`)

## Important Notes

- **Optional dependencies are extensive**: 20+ extra groups. Most policy/robot code requires specific extras. The `[all]` extra excludes `wallx`, `pi`, and `groot` due to `transformers` version conflicts.
- **`uv` conflicts**: wallx requires `transformers==4.49.0` while others need `>=4.53.0`, declared in `[tool.uv] conflicts` in pyproject.toml.
- **Git LFS required**: Test artifacts (`.safetensors`, `.mp4`, `.arrow`, `.png` in `tests/artifacts/`) are in Git LFS. Tests fail without `git lfs pull`.
- **`MUJOCO_GL=egl`** is needed for headless simulation environments.
- **Test device**: Controlled by `LEROBOT_TEST_DEVICE` env var (defaults to `cuda` if available, else `cpu`).
- **Hardware test mocks**: `tests/mocks/` provides mock motor buses, robots, and serial ports for CI.
- **Factory lazy imports**: `lerobot/policies/factory.py` uses lazy imports to avoid loading all dependencies. Follow this pattern when adding policies.
