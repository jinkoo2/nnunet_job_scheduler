
import json, os, re
from pathlib import Path

from logger import log, log_exception as LE, log_and_raise_exception as LER

from config import config

nnunet_preprocessed_dir =  config['preprocessed_dir']

nnunet_planner = config['nnunet_planner']

slurm_user = config['slurm_user']
slurm_email = config['slurm_email']

slurm_num_of_tasks_per_node = config['slurm_num_of_tasks_per_node']
slurm_num_of_nodes = config['slurm_num_of_nodes']
slurm_num_of_hours = config['slurm_num_of_hours']
slurm_partition = config['slurm_partition']
slurm_num_of_gpus_per_node = config['slurm_num_of_gpus_per_node']

 
log(f'preprocessed_dir={nnunet_preprocessed_dir}')

import raw

from utils import path_found

def id_list():
    """Get a list of data set."""
    pattern = r"^Dataset\d{3}_.+$"  # Regex for Datasetxxx_yyyyyy format
    return [entry.name for entry in Path(nnunet_preprocessed_dir).iterdir() if entry.is_dir() and re.match(pattern, entry.name)]

def get_dataset_id_list():
    return id_list()

def get_dataset_num_list():
    id_list = get_dataset_id_list()
    num_list = [re.search(r'Dataset(\d+)_', name).group(1) for name in id_list]
    return num_list

def get_complated_dataset_id_list():
    return [id for id in id_list() if complated(id)]

def case_dir(id):
    return os.path.join(nnunet_preprocessed_dir, id)
def case_dir_exists(id):
    return path_found(case_dir(id))

def plan_json_file(id):
    return os.path.join(nnunet_preprocessed_dir, id, 'nnUNetPlans.json')
def plan_json_exists(id):
    return path_found(plan_json_file(id))
    
def plan_json(id):
    with open(plan_json_file(id), 'r') as f:
        data = json.load(f)
    return data

def plan_conf_dir_list(id):
    confs = plan_json(id)['configurations']
    list = []
    for conf_key in confs.keys():
        if 'cascade' in conf_key:
            continue
        data_identifier = confs[conf_key]['data_identifier']
        conf_dir = os.path.join(case_dir(id), data_identifier)
        list.append(conf_dir)
    return list

def conf_dir(id):
    if is_2d(id):
        return os.path.join(case_dir(id), 'nnUNetPlans_2d')
    else:
        return os.path.join(case_dir(id), 'nnUNetPlans_3d_lowres')

def conf_dir_exists(id):
    
    if not case_dir_exists(id)['exists']:
        return {'exists': False, 'reason': f'case_dir not found - {case_dir(id)}'}
    
    return path_found(conf_dir(id))

def plan_conf_list(id):
    confs = plan_json(id)['configurations']
    return list(confs.keys())

def is_2d(id):
    confs = plan_conf_list(id)
    return len(confs) ==1 and confs[0] == '2d'

def all_processed_images_exist(id):

    if not conf_dir_exists(id)['exists']:
        return conf_dir_exists(id)
    
    # training image file ids from raw
    raw_train_image_ids = raw.images_tr_file_id_list(id)
    if len(raw_train_image_ids) == 0:
        return {
            'exists':False,
            'reason':f'there is no training image in raw_dir for {id}'
        } 

    # get all files 
    files = os.listdir(conf_dir(id))

    # check there are at least two processed images for each image_id
    for image_id in raw_train_image_ids:
        matching_files = [f for f in files if image_id in f]
        if len(matching_files) < 2:
            return {
                'exists':False,
                'reason':f'Less than 2 files found for image id [{image_id}] in the configuration folder: {conf_dir}]'
            }
    
    return {
                'exists':True,
                'reason':''
            }

def dataset_json_file(id):
    return os.path.join(nnunet_preprocessed_dir, id, 'dataset.json')
def dataset_json_exists(id):
    return path_found(dataset_json_file(id))
def dataset_json(id):
    with open(dataset_json_file(id), 'r') as f:
        data = json.load(f)
    return data

def complated(id):
    s = status(id)
    for key in s.keys():
        if 'exists' in s[key] and not s[key]['exists']:
            return False
    return True


def status(id):

    if not case_dir_exists(id)['exists']:
        return {
            "case_dir_exists": case_dir_exists(id),
            "nnUNetPlans_json_exists": {'exists':False, 'reason':''},
            "dataset_json_exists": {'exists':False, 'reason':''},
            "conf_dir_exists": {'exists':False, 'reason':''},
            "all_processed_images_exist_in_conf_folders": {'exists':False, 'reason':''}
        }
    else:
        import utils
        return {
            "case_dir_exists": case_dir_exists(id),
            "nnUNetPlans_json_exists": plan_json_exists(id),
            "dataset_json_exists": dataset_json_exists(id),
            "conf_dir_exists": conf_dir_exists(id),
            "all_processed_images_exist": all_processed_images_exist(id),
            #"files": utils.list_files(case_dir(id))
        }

def submit_slurm_job(id):
    
    job_num = id[7:10]

    job_name = f'pp_{job_num}'

    log(f'checking job {job_name} is already in the queue or running')
    from simple_slurm_server import slurm_commands
    jobs = slurm_commands.get_jobs_of_user(slurm_user)
    
    jobs_of_name = [job for job in jobs if job['name'] == job_name]

    if len(jobs_of_name) > 0:
        log(f'job is already in the queue ')
        job = jobs_of_name[0]
        log(json.dumps(job, indent=4))
        return 
    
    venv_dir = config['venv_dir']
    nnunet_dir = config['nnunet_dir']
    raw_dir = config['raw_dir']
    preprocessed_dir = config['preprocessed_dir']
    results_dir = config['results_dir']

    job_name = f'pp_{job_num}'
    partition = slurm_partition
    num_of_nodes = slurm_num_of_nodes
    ntasks_per_node = slurm_num_of_tasks_per_node
    num_of_gpus_per_node = slurm_num_of_gpus_per_node
    num_of_hours = slurm_num_of_hours
    email = slurm_email
    dataset_num = job_num
    planner= nnunet_planner

    configuration = '2d' if is_2d(id) else '3d_lowres'
    
    cmd_line = f'nnUNetv2_plan_and_preprocess -d {dataset_num} -pl {planner} -c {configuration} -npfp 1 -np 1 --verbose --verify_dataset_integrity '
    
    script_output_files_dir = config['script_output_files_dir']
    case_scripts_dir = os.path.join(script_output_files_dir, job_num)
    if not os.path.exists(case_scripts_dir):
        os.makedirs(case_scripts_dir)

    ### script file
    script_file = os.path.join(case_scripts_dir, f'pp_{job_num}.slurm')

    ### log file
    log_file = script_file+'.log'
    if os.path.exists(log_file):
        log(f'previous log file found. Removing it.... {log_file}')
        try:
            os.remove(log_file)
        except FileNotFoundError:
            log(f"File {log_file} not found.")
        except PermissionError:
            log(f"Permission denied to delete {log_file}.")
        except OSError as e:
            log(f"Error deleting {log_file}: {e}")

    slurm_head = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_file}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --nodes={num_of_nodes}
#SBATCH --time={num_of_hours}:00:00
#SBATCH -p {partition}
#SBATCH --gres=gpu:{num_of_gpus_per_node}            # number of GPUs per node (gres=gpu:N)
#SBATCH --gpus={num_of_gpus_per_node} # number of GPUs per node (gres=gpu:N), this is a backup of --gres
#SBATCH --mail-type=ALL
#SBATCH --mail-user={email}
'''

    script = f'''{slurm_head}

source {venv_dir}/bin/activate
cd {nnunet_dir}

export nnUNet_raw="{raw_dir}"
export nnUNet_preprocessed="{preprocessed_dir}"
export nnUNet_results="{results_dir}"

{cmd_line}
'''
    
    log(f'script={script}')

    log(f'saving script file - {script_file}')
    with open(script_file, 'w') as file:
        file.write(script)


    cmd = f'module load slurm && sbatch {script_file}'
    log(f'running "{cmd}"')
    import subprocess
    subprocess.run(cmd, shell=True)

    from simple_slurm_server import slurm_commands

    
    log(f'pp job submitted: {job_num}')

    jobs = slurm_commands.get_jobs_of_user(slurm_user)

    log(f'slurm jobs')
    log(json.dumps(jobs, indent=4))
    

def check_and_submit_pp_jobs():
    log('******** dataset ready for pp *********************')
    id_list = raw.dataset_id_list_ready_for_pp()

    # get pp status list
    id_list_to_pp = []
    for id in id_list:
        log(f'=== {id} ===')
        if complated(id):
            log(f'\t{id} - Completed')
        else:
            log(f'\t{id} - NOT completed')
            log(json.dumps(status(id), indent=4))
            id_list_to_pp.append(id)
            
    # submit pp jobs
    log('****submititng pp jobs****')
    for id in id_list_to_pp:
        log(f'submiting slurm job for {id}')
        submit_slurm_job(id)
        

if __name__ == '__main__':
    check_and_submit_pp_jobs()
    log('done')




