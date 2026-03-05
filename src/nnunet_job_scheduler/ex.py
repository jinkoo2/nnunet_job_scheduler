"""
Model export and dashboard upload.

After training completes for a dataset, this module:
1. Runs nnUNetv2_export_model_to_zip to produce a ZIP of the trained model.
2. Registers the scheduler as a worker with the dashboard (upsert by name).
3. Looks up the dataset by name in the dashboard to get its UUID.
4. Creates a training_job record in the dashboard.
5. Uploads the model ZIP to the dashboard.
6. Writes a local flag file so the dataset is not uploaded again.
"""
import os
import re
import subprocess
from pathlib import Path

from nnunet_job_scheduler.logger import log, log_exception
from nnunet_job_scheduler.config import config
from nnunet_job_scheduler import tr, raw

UPLOADED_FLAG = 'dashboard_uploaded.txt'


def _get_dataset_num(id):
    m = re.search(r'Dataset(\d+)_', id)
    if not m:
        raise ValueError(f"Cannot extract dataset number from: {id!r}")
    return str(int(m.group(1)))


def _configuration_for(id):
    return '2d' if raw.is_2d(id) else '3d_lowres'


def _uploaded_flag_path(id):
    return os.path.join(config['results_dir'], id, UPLOADED_FLAG)


def is_model_uploaded(id):
    return os.path.exists(_uploaded_flag_path(id))


def _mark_uploaded(id, configuration, model_id):
    flag = _uploaded_flag_path(id)
    os.makedirs(os.path.dirname(flag), exist_ok=True)
    with open(flag, 'w') as f:
        f.write(f"configuration={configuration}\nmodel_id={model_id}\n")


def export_model(id, configuration):
    """
    Export trained model to ZIP using nnUNetv2_export_model_to_zip.
    Returns the path to the created ZIP file.
    """
    dataset_num = _get_dataset_num(id)
    venv_dir = config['venv_dir']
    raw_dir = config['raw_dir']
    preprocessed_dir = config['preprocessed_dir']
    results_dir = config['results_dir']

    output_zip = os.path.join(results_dir, id, 'model.zip')
    if os.path.exists(output_zip):
        os.remove(output_zip)

    cmd = (
        f'source "{venv_dir}/bin/activate" && '
        f'export nnUNet_raw="{raw_dir}" && '
        f'export nnUNet_preprocessed="{preprocessed_dir}" && '
        f'export nnUNet_results="{results_dir}" && '
        f'nnUNetv2_export_model_to_zip -d {dataset_num} -c {configuration} -o "{output_zip}" --not_strict'
    )

    log(f'Exporting model: {id} ({configuration}) -> {output_zip}')
    result = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Model export failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    if not os.path.exists(output_zip):
        raise RuntimeError(f"Export succeeded but ZIP not found at {output_zip}")

    size_mb = os.path.getsize(output_zip) / 1024 / 1024
    log(f'Model exported: {output_zip} ({size_mb:.1f} MB)')
    return output_zip


def upload_to_dashboard(id, configuration, zip_path):
    """
    Upload model zip to dashboard.
    Registers the scheduler as a worker, looks up dataset by name,
    creates a job, and uploads the model.
    Returns the dashboard model_id.
    """
    from nnunet_job_scheduler.dashboard_client import DashboardClient
    client = DashboardClient()

    worker_name = config.get('dashboard_worker_name', 'nnunet_job_scheduler')

    log(f'Registering with dashboard as worker: {worker_name}')
    worker = client.register_worker(worker_name)
    worker_id = worker['id']
    log(f'Worker ID: {worker_id}')

    log(f'Looking up dataset "{id}" in dashboard...')
    datasets = client.list_datasets()
    dashboard_dataset = next((ds for ds in datasets if ds.get('name') == id), None)
    if dashboard_dataset is None:
        raise RuntimeError(
            f"Dataset '{id}' not found in dashboard. "
            "Upload the dataset to the dashboard first before exporting the model."
        )
    dataset_id = dashboard_dataset['id']
    log(f'Dashboard dataset_id: {dataset_id}')

    log(f'Creating training job in dashboard (configuration={configuration})...')
    job = client.create_job(dataset_id, worker_id, configuration)
    job_id = job['id']
    log(f'Dashboard job_id: {job_id}')

    client.update_job_status(job_id, 'uploading')

    log(f'Uploading model zip...')
    model = client.upload_model(job_id, zip_path)
    model_id = model['id']

    client.update_job_status(job_id, 'done')

    log(f'Model uploaded to dashboard: model_id={model_id}')
    return model_id


def check_and_export_models():
    """
    Find all training-completed datasets and upload their models to the dashboard.
    Skips datasets that have already been uploaded (flag file present).
    Skips silently if dashboard_url is not configured.
    """
    dashboard_url = config.get('dashboard_url', '')
    if not dashboard_url:
        log('dashboard_url not configured — skipping model export/upload.')
        return

    log('Looking for training-completed datasets to export and upload...')
    completed_ids = tr.get_completed_dataset_id_list()
    log(f'Training-completed datasets: {completed_ids}')

    for id in completed_ids:
        if is_model_uploaded(id):
            log(f'{id} — already uploaded to dashboard, skipping.')
            continue

        configuration = _configuration_for(id)
        log(f'{id} — exporting model (configuration={configuration})...')
        try:
            zip_path = export_model(id, configuration)
            model_id = upload_to_dashboard(id, configuration, zip_path)
            _mark_uploaded(id, configuration, model_id)
            log(f'{id} — model uploaded successfully (model_id={model_id})')
        except Exception as e:
            log(f'{id} — export/upload failed: {e}')
            log_exception(e)
