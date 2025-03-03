import os

#################
# ENV load .env file
from dotenv import load_dotenv
load_dotenv()

from utils import _error, _info, _join_dir

def get_config():
   
    home_dir = os.getenv('home_dir')
    venv_dir = os.getenv('venv_dir')
    script_output_files_dir = os.getenv('script_output_files_dir')
    nnunet_dir = os.getenv('nnunet_dir')
    data_dir = os.getenv('data_dir')
    log_dir = os.getenv('log_dir')

    if not os.path.exists(nnunet_dir):
        _error(f'nnunet_dir not found:{nnunet_dir}')

    if not os.path.exists(data_dir):
        _error(f'data_dir not found:{data_dir}')

    raw_dir = _join_dir(data_dir, 'raw')
    preprocessed_dir= _join_dir(data_dir, 'preprocessed')
    results_dir = _join_dir(data_dir, 'results')


    ret = {
        'home_dir': home_dir,
        'venv_dir': venv_dir,
        'nnunet_dir': nnunet_dir,
        'data_dir': data_dir,
        'raw_dir': raw_dir,
        'preprocessed_dir': preprocessed_dir,
        'results_dir': results_dir,
        'script_output_files_dir': script_output_files_dir,
        'log_dir': log_dir
    }

    #print('get_config().return=', ret)

    return ret
