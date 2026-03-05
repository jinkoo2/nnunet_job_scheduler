# nnunet_job_scheduler

Autonomous nnUNet job scheduler that monitors the filesystem and submits SLURM jobs for preprocessing, training, and model export/upload.

## Overview

The scheduler runs as a background process and on each cycle (every 60 minutes):

1. **Preprocessing** (`pp`) — finds datasets in `raw/` that are ready, checks if preprocessing is complete, and submits `nnUNetv2_plan_and_preprocess` SLURM jobs for any that are not.
2. **Training** (`tr`) — finds preprocessing-complete datasets, checks which folds are missing checkpoints, and submits `nnUNetv2_train` SLURM jobs.
3. **Export & Upload** (`ex`) — finds training-complete datasets (all 5 folds done), exports the model to a ZIP with `nnUNetv2_export_model_to_zip`, and uploads it to the nnunet-dashboard. A flag file (`dashboard_uploaded.txt`) prevents duplicate uploads.

## Setup

```bash
# Create and activate the venv
conda activate nnunet_trainer   # or use the project venv

# Install dependencies
pip install -e .
# or
poetry install

# Configure
cp .env.example .env   # if available, or edit .env directly
```

## Configuration (`.env`)

| Variable | Description |
|---|---|
| `venv_dir` | Path to the Python venv with nnunetv2 installed |
| `nnunet_dir` | Path to the nnUNet source directory |
| `data_dir` | Root data directory (contains `raw/`, `preprocessed/`, `results/`) |
| `script_output_files_dir` | Directory for generated SLURM scripts and logs |
| `log_dir` | Directory for scheduler log files |
| `case_status_list_dir` | Directory for per-dataset status JSON files |
| `nnunet_trainer` | nnUNet trainer class (default: `nnUNetTrainer`) |
| `nnunet_planner` | nnUNet planner class (default: `ExperimentPlanner`) |
| `nnunet_plans` | nnUNet plans name (default: `nnUNetPlans`) |
| `slurm_user` | SLURM username for job queue checks |
| `slurm_email` | Email for SLURM job notifications |
| `slurm_partition` | SLURM partition for training jobs |
| `slurm_num_of_nodes` | Number of nodes per job |
| `slurm_num_of_tasks_per_node` | Tasks per node |
| `slurm_num_of_hours` | Wall time limit in hours |
| `slurm_num_of_gpus_per_node` | GPUs per node |
| `slurm_max_jobs_per_user` | Max concurrent SLURM jobs (default: `10`) |
| `min_num_of_required_training_images` | Minimum images required to start preprocessing |
| `dashboard_url` | nnunet-dashboard base URL (e.g. `https://nnunet-dashboard-1.apps.myphysics.net/`) |
| `dashboard_api_key` | API key for the dashboard (`X-Api-Key` header) |
| `dashboard_worker_name` | Worker name to register with the dashboard (default: `nnunet_job_scheduler`) |

## Running

```bash
poetry run main
```

Or directly:

```bash
python -m nnunet_job_scheduler.app
```

## Dashboard Integration

When training is complete for a dataset, the scheduler:

1. Registers itself as a worker on the dashboard (upsert by `dashboard_worker_name`).
2. Looks up the dataset by name (e.g. `Dataset289_SRSBody`) in the dashboard.
3. Creates a training job record and uploads the model ZIP.
4. Marks the job as `done` in the dashboard.
5. Writes `results/{dataset}/dashboard_uploaded.txt` to prevent re-uploading.

The dataset must already exist in the dashboard (uploaded via `nnunet_server` or the dashboard UI) for the upload to succeed. If not found, the export step is skipped with a warning.

## Data Layout

```
data_dir/
├── raw/Dataset###_Name/            ← raw images + dataset.json
├── preprocessed/Dataset###_Name/  ← nnUNetPlans.json + preprocessed data
├── results/Dataset###_Name/       ← trained models (managed by nnUNet)
│   └── dashboard_uploaded.txt     ← flag: model already uploaded to dashboard
└── predictions/                   ← inference outputs
```

## Key Files

| File | Description |
|---|---|
| `src/nnunet_job_scheduler/app.py` | Main loop |
| `src/nnunet_job_scheduler/config.py` | Config loaded from `.env` |
| `src/nnunet_job_scheduler/pp.py` | Preprocessing logic and SLURM submission |
| `src/nnunet_job_scheduler/tr.py` | Training logic and SLURM submission |
| `src/nnunet_job_scheduler/ex.py` | Model export and dashboard upload |
| `src/nnunet_job_scheduler/dashboard_client.py` | HTTP client for the nnunet-dashboard API |
| `src/nnunet_job_scheduler/raw.py` | Raw dataset utilities |
| `src/nnunet_job_scheduler/slurm.py` | SLURM job listing utilities |
