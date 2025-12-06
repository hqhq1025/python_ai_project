# Repository Guidelines

用中文跟我交流
参考 draft 的思想, 在和我的讨论中形成一个 plan.md
确定好 plan.md 后, 按照 plan.md 进行执行
每次完成一个任务或者更新去求时, 请同步更新 plan.md

This document is a concise guide for contributors working in this repository, including human and AI agents.

## Project Structure & Module Organization

- Place production code under `src/` using a package such as `src/project_name/`.
- Put tests in `tests/` mirroring the `src/` layout (e.g., `src/project_name/model.py` → `tests/test_model.py`).
- Use `notebooks/` for exploratory work and keep them lightweight and reproducible.
- Store large or generated artifacts in `data/` or `outputs/` and avoid committing anything that can be regenerated.

## Build, Test, and Development Commands

- `python -m pip install -e .[dev]` – install the project in editable mode with development dependencies.
- `python -m pytest` – run the full test suite.
- `python -m pytest tests/test_x.py -k pattern` – run a focused subset of tests.
- `python -m project_name` or a dedicated CLI script – run the main application entry point (if defined).

## Coding Style & Naming Conventions

- Use Python 3, 4-space indentation, and type hints for all new or modified functions.
- Follow PEP 8: `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Prefer small, focused modules; avoid large, monolithic files.
- If formatters/linters are configured (e.g., `black`, `ruff`, `flake8`), run them before opening a PR.

## Testing Guidelines

- Write tests for all new behavior and bug fixes; prefer unit tests close to the relevant module.
- Name test files as `tests/test_<module>.py` and test functions as `test_<behavior>`.
- Keep tests deterministic and isolated from network, filesystem, and external services when possible.

## Commit & Pull Request Guidelines

- Use clear, imperative commit messages (e.g., `Add dataset loader`, `Fix inference bug`).
- Keep PRs focused and small; include a short summary of changes, rationale, and any breaking impacts.
- Link related issues and include usage examples or screenshots for user-facing changes.

## Agent-Specific Instructions

- When editing files, keep changes minimal and localized to the task.
- Do not introduce new external dependencies or tools without clear justification in PR descriptions.
- Respect this guide for any code or documentation you create or modify in this repository.

