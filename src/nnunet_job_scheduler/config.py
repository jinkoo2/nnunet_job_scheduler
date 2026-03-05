import os

#################
# ENV load .env file
from dotenv import load_dotenv
load_dotenv()

from nnunet_job_scheduler.utils import _error, _info, _join_dir

def get_config():
    # home_dir = os.getenv('home_dir')  # legacy, unused
    venv_dir = os.getenv('venv_dir')
    script_output_files_dir = os.getenv('script_output_files_dir')
    nnunet_dir = os.getenv('nnunet_dir')
    data_dir = os.getenv('data_dir')
    log_dir = os.getenv('log_dir')
    case_status_list_dir = os.getenv('case_status_list_dir')

    nnunet_trainer = os.getenv('nnunet_trainer')
    nnunet_planner = os.getenv('nnunet_planner')
    nnunet_plans = os.getenv('nnunet_plans')

    slurm_user = os.getenv('slurm_user')
    slurm_email = os.getenv('slurm_email')
    slurm_num_of_tasks_per_node = int(os.getenv('slurm_num_of_tasks_per_node'))
    slurm_num_of_nodes = int(os.getenv('slurm_num_of_nodes'))
    slurm_num_of_hours = int(os.getenv('slurm_num_of_hours'))
    slurm_partition = os.getenv('slurm_partition')
    slurm_max_jobs_per_user = int(os.getenv('slurm_max_jobs_per_user', '10'))

    dashboard_url = os.getenv('dashboard_url', '')
    dashboard_api_key = os.getenv('dashboard_api_key', '')
    dashboard_worker_name = os.getenv('dashboard_worker_name', 'nnunet_job_scheduler')
    enable_export_to_zip = os.getenv('enable_export_to_zip', 'true').strip().lower() == 'true'
    enable_upload_to_dashboard = os.getenv('enable_upload_to_dashboard', 'false').strip().lower() == 'true'

    slurm_num_of_gpus_per_node = int(os.getenv('slurm_num_of_gpus_per_node'))
    

    min_num_of_required_training_images = int(os.getenv('min_num_of_required_training_images'))

    if not os.path.exists(nnunet_dir):
        _error(f'nnunet_dir not found:{nnunet_dir}')

    if not os.path.exists(data_dir):
        _error(f'data_dir not found:{data_dir}')

    # Ensure key directories exist
    if script_output_files_dir:
        os.makedirs(script_output_files_dir, exist_ok=True)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    if case_status_list_dir:
        os.makedirs(case_status_list_dir, exist_ok=True)

    slurm_jobs_dir = _join_dir(data_dir, 'slurm_jobs_dir')

    raw_dir = _join_dir(data_dir, 'raw')
    preprocessed_dir= _join_dir(data_dir, 'preprocessed')
    results_dir = _join_dir(data_dir, 'results')
    predictions_dir = _join_dir(data_dir, 'predictions')

    ret = {
        # 'home_dir': home_dir,  # legacy, unused
        'venv_dir': venv_dir,
        'nnunet_dir': nnunet_dir,
        'data_dir': data_dir,
        'raw_dir': raw_dir,
        'preprocessed_dir': preprocessed_dir,
        'results_dir': results_dir,
        'predictions_dir': predictions_dir,
        'nnunet_trainer': nnunet_trainer,
        'nnunet_plans' : nnunet_plans,
        'nnunet_planner': nnunet_planner,
        'script_output_files_dir': script_output_files_dir,
        'log_dir': log_dir,
        'slurm_user': slurm_user,
        'slurm_email': slurm_email,
        'slurm_num_of_tasks_per_node':slurm_num_of_tasks_per_node,
        'slurm_num_of_nodes':slurm_num_of_nodes,
        'slurm_num_of_hours':slurm_num_of_hours,
        'slurm_partition': slurm_partition,
        'slurm_max_jobs_per_user': slurm_max_jobs_per_user,
        'case_status_list_dir': case_status_list_dir,
        'slurm_jobs_dir':slurm_jobs_dir,
        'slurm_num_of_gpus_per_node':slurm_num_of_gpus_per_node,
        'min_num_of_required_training_images': min_num_of_required_training_images,
        'dashboard_url': dashboard_url,
        'dashboard_api_key': dashboard_api_key,
        'dashboard_worker_name': dashboard_worker_name,
        'enable_export_to_zip': enable_export_to_zip,
        'enable_upload_to_dashboard': enable_upload_to_dashboard,
    }

    return ret

config = get_config()