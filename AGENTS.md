# Repository Guidelines

## Project Structure & Module Organization
The repo is split between source, assets, and experiments. Place all production Python under `src/fitness_ai/` following the package map from the README (`data/`, `models/`, `training/`, `utils/`). Raw sensor dumps live in `data/raw/` (e.g., `run.csv`), while filtered artifacts belong in `data/processed/`. Keep serialized weights or TensorFlow Lite exports inside `models/` so they stay versioned yet isolated. Exploratory work, plots, and diagnostics remain in `notebooks/` (see `LowPassFilter.ipynb`) and should only read from `data/processed` to avoid overwriting raw inputs.

## Build, Test, and Development Commands
Use Conda to reproduce the base toolchain: `conda env create -f environment.yml` and `conda activate Machine-Learning-for-Smart-Fitness-Pod`. Install the package for local imports with `pip install -e .`, which exposes `fitness_ai` modules inside notebooks and scripts. Launch analyses via `jupyter lab notebooks/LowPassFilter.ipynb` or similar. When you add automation scripts (trainers, converters), prefer `python -m fitness_ai.training.trainer` so modules resolve correctly.

## Coding Style & Naming Conventions
Code is Python-first; follow PEP 8, 4-space indentation, and descriptive snake_case filenames (`preprocessor.py`, `feature_extractor.py`). Class names use CapWords (`ExerciseClassifier`), and experiment notebooks should start with verbs (`LowPassFilter.ipynb`). Keep functions small, pure, and type-annotated when IO boundaries touch model code. Run `ruff` or `black` if available in your editor; otherwise, ensure imports are grouped stdlib/third-party/local and keep docstrings factual.

## Testing Guidelines
Automated tests belong in `tests/` mirroring the package tree (e.g., `tests/training/test_trainer.py`). Use `pytest` naming (`test_*`) and sample CSV fixtures from `data/processed`. Minimum expectation is to validate preprocessing logic and unit-test model wrappers before pushing. Run `python -m pytest` locally and mention any skipped cases in the PR. For notebook-driven checks, save generated metrics under `notebooks/reports/` and summarize the threshold you validated (accuracy, repetition error).

## Commit & Pull Request Guidelines
Commits should stay concise and imperative, following the existing history (`change the path of run.csv`, `Construct the project so ...`). Reference issue IDs when available and keep scope to one feature or fix. PRs must include: purpose, key files touched, how to reproduce (`python -m pytest`, `jupyter lab ...`), and screenshots or metric tables for model improvements. Request at least one review before merging and ensure large data files remain ignored via `.gitignore`.

## Security & Data Handling
Never commit private keys or sensitive sensor dumps; scrub metadata before sharing. If logs must ship, stage them under `data/processed/` with anonymized ids and update `.gitignore` when adding new capture directories.
