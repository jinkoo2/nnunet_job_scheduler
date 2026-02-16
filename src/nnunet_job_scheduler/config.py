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

    nnunet_trainer = os.getenv('nnunet_trainer')
    nnunet_planner = os.getenv('nnunet_planner')
    nnunet_plans = os.getenv('nnunet_plans')

    slurm_user = os.getenv('slurm_user')
    slurm_email = os.getenv('slurm_email')
    slurm_num_of_tasks_per_node = int(os.getenv('slurm_num_of_tasks_per_node'))
    slurm_num_of_nodes = int(os.getenv('slurm_num_of_nodes'))
    slurm_num_of_hours = int(os.getenv('slurm_num_of_hours'))
    slurm_partition = os.getenv('slurm_partition')

    slurm_jobs_dir = _join_dir(data_dir, 'slurm_jobs_dir')

    slurm_num_of_gpus_per_node = int(os.getenv('slurm_num_of_gpus_per_node'))
    

    min_num_of_required_training_images = int(os.getenv('min_num_of_required_training_images'))

    if not os.path.exists(nnunet_dir):
        _error(f'nnunet_dir not found:{nnunet_dir}')

    if not os.path.exists(data_dir):
        _error(f'data_dir not found:{data_dir}')

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
        'slurm_jobs_dir':slurm_jobs_dir,
        'slurm_num_of_gpus_per_node':slurm_num_of_gpus_per_node,
        'min_num_of_required_training_images': min_num_of_required_training_images
    }

    return ret

config = get_config()