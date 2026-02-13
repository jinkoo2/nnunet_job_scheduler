# AGENTS.md – nnunet_job_scheduler

Guidance for AI agents working on this project.

---

## Project overview

**nnunet_job_scheduler** is a long-running daemon that monitors nnUNet v2 datasets on the filesystem and submits **Slurm** jobs for the nnUNet pipeline: **plan & preprocess (pp)** → **training (tr)** → **prediction (pr)**. It runs in a loop (default every 60 minutes), checks completion state per dataset/case, and avoids duplicate submissions by checking existing Slurm jobs.

- **Entry point**: `poetry run main` → `nnunet_job_scheduler.app:main`
- **Config**: Environment variables loaded from `.env` via `python-dotenv` (see [Configuration](#configuration)).
- **Slurm**: In-package `slurm_commands` module (runs `module load slurm` then squeue/scontrol/scancel) for job queries and submission; no external Slurm library dependency.

---

## Architecture and pipeline

1. **raw** – Raw nnUNet datasets under `data_dir/raw/`. Dataset IDs match `Dataset\d{3}_.+`. Each case has `dataset.json`, `imagesTr`, `labelsTr`, etc. Used to decide if a dataset is ready for planning/preprocessing.
2. **pp** – Plan & preprocess. Reads/writes under `data_dir/preprocessed/`. Submits `nnUNetv2_plan_and_preprocess` when a raw dataset is ready and not yet completed.
3. **tr** – Training. Reads/writes under `data_dir/results/`. Submits `nnUNetv2_train` per dataset and fold (0–4). “Completed” means all folds have checkpoints and validation summary.
4. **pr** – Prediction. Reads/writes under `data_dir/predictions/`. Each case has `req_001`, `req_002`, … with inputs and `outputs/`. Submits `nnUNetv2_predict` when training is complete and not all inputs for a request have outputs.

The main loop in `app.main()` runs, in order: `pp.check_and_submit_pp_jobs()`, `tr.check_and_submit_tr_jobs()`, `pr.check_and_submit_pr_jobs()`, then `slurm.save_jobs_of_user()`, then sleeps 60 minutes.

---

## Directory layout (source)

```
projects/nnunet_job_scheduler/
├── pyproject.toml          # Project name: nnunet-jobs-handler; script: main → app:main
├── README.md
├── AGENTS.md               # This file
└── src/nnunet_job_scheduler/
    ├── app.py              # main() – loop: pp → tr → pr → slurm save → sleep
    ├── config.py           # get_config() from env; config = get_config()
    ├── logger.py           # log(), log_exception(), log_and_raise_exception(); daily log files in log_dir
    ├── utils.py            # path_found(), file_id_list(), list_files(), _join_dir(), etc.
    ├── raw.py              # Raw dataset listing and status; dataset.json; pp_ready(); is_2d/is_3d
    ├── pp.py               # Preprocessed status; submit pp Slurm jobs
    ├── tr.py               # Training results, folds, checkpoints; submit tr Slurm jobs
    ├── pr.py               # Predictions, req_* dirs; submit pr Slurm jobs
    ├── slurm_commands.py   # run_command(), get_jobs(), get_jobs_of_user(), get_job_from_job_name(), cancel_job(), etc.
    ├── slurm.py            # save_jobs_of_user(), save_jobs_of_all_users()
    └── slurm/
        ├── list_my_jobs.py   # CLI: list current user’s Slurm jobs
        └── cancel_my_jobs.py # CLI: cancel current user’s Slurm jobs
```

Generated at runtime (from config):

- `script_output_files_dir/<dataset_num>/*.slurm` – generated Slurm scripts
- `log_dir/` – daily logs (e.g. `YYYY-MM-DD.log`)
- `slurm_jobs_dir/` – saved job JSON (e.g. `jobs.<node>.<user>.json`)

---

## Configuration

All configuration comes from environment variables (e.g. `.env`). Required keys include:

- **Paths**: `home_dir`, `venv_dir`, `script_output_files_dir`, `nnunet_dir`, `data_dir`, `log_dir`
- **nnUNet**: `nnunet_trainer`, `nnunet_planner`, `nnunet_plans`, `min_num_of_required_training_images`
- **Slurm**: `slurm_user`, `slurm_email`, `slurm_num_of_tasks_per_node`, `slurm_num_of_nodes`, `slurm_num_of_hours`, `slurm_partition`, `slurm_num_of_gpus_per_node`

Derived paths in config: `raw_dir`, `preprocessed_dir`, `results_dir`, `predictions_dir`, `slurm_jobs_dir` (under `data_dir`). Do not hardcode paths; use `config` from `config.py`.

---

## Conventions and patterns

- **Dataset/case ID**: String like `Dataset015_SomeName`. Dataset number is `id[7:10]` (e.g. `015`). Used in job names: `pp_015`, `tr_015_0`, `pr015057`.
- **Status helpers**: Many modules have `status(id)` returning a dict of `{..., 'exists': bool, 'reason': ...}`. Completion is checked with `complated(id)` (see below).
- **Intentional typo**: The codebase uses `complated` (not “completed”) in function names, e.g. `complated(id)`, `get_complated_dataset_id_list()`. Preserve this naming when adding or refactoring to avoid breaking call sites.
- **Slurm script generation**: Each stage builds a bash script with `#SBATCH` headers and `source venv; cd nnunet_dir; export nnUNet_*; <nnUNet command>`, then runs `module load slurm && sbatch <script_file>`.
- **Logging**: Use `from nnunet_job_scheduler.logger import log, log_exception` and call `log(msg)` or `log_exception(e)`; do not add new log files or formatters unless required.

---

## Known issues and fragile spots

- **pr.py**
  - `re` is used in `id_list()` but `import re` appears later in the file; move `import re` to the top.
  - `req_num_list()` uses `return nums(sorted)`; should be `return sorted(nums)`.
  - `status(id)` has a malformed `if/else`: the `else` block is not correctly paired with the `if not case_dir_exists(id)['exists']` (syntax/flow bug).
- **raw.py**
  - In `status(id)`, one branch calls `dataset_json_exists(id, 'dataset.json')` but `dataset_json_exists(id)` only takes one argument; this will raise at runtime when that branch is taken.
- **utils.py**
  - `_error(msg)` raises `Exception(f'Error: msg')` so the argument `msg` is not interpolated; it should be `Exception(msg)` or `Exception(f'Error: {msg}')`.
- **tr.py**
  - Log message says "pp jost submitted" (typo); "pr jost" appears in pr.py as well – minor.
  - `get_config()` is called in `submit_slurm_job` while most code uses the global `config`; consistent use of `config` is preferable.

When editing these areas, fix the bug you are targeting and avoid unnecessary renames (e.g. `complated` → `completed`) unless the user explicitly asks for a project-wide rename.

---

## Running and testing

- **Install**: From project root, `poetry install` (uses `pyproject.toml`; depends on `simple-slurm-server` and `python-dotenv`).
- **Run daemon**: `poetry run main` (or activate venv and run `main`). Ensure `.env` is set and `data_dir`/nnUNet paths exist.
- **One-off checks**: You can run `pp.main()`, `tr.check_and_submit_tr_jobs()`, or `pr.check_and_submit_pr_jobs()` from a shell or test for debugging; the app normally runs them in sequence in a loop.
- **Slurm**: Scripts assume `module load slurm` and `sbatch` are available; job names and paths are tied to the configured `slurm_user` and directories.

---

## Summary for agents

- This is a **nnUNet v2** + **Slurm** scheduler daemon: **pp** → **tr** → **pr**, with config from `.env`.
- Key modules: `app` (loop), `config`, `raw`, `pp`, `tr`, `pr`, `slurm`, `utils`, `logger`.
- Keep the existing naming (`complated`, job names, path conventions) and fix only the bugs listed above when relevant.
- Add new features (e.g. new pipeline stages or checks) by following the same patterns: `status(id)`-like dicts, `path_found()`, and the existing Slurm script generation style.
