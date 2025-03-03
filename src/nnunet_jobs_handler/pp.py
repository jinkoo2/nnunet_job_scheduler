
import json
import os
import re
from pathlib import Path

from logger import logger, log_exception as LE, log_and_raise_exception as LER

from config import get_config
nnunet_data_dir = get_config()["data_dir"] 
nnunet_raw_dir =  os.path.join(nnunet_data_dir,'raw')
nnunet_preprocessed_dir =  os.path.join(nnunet_data_dir,'preprocessed')
nnunet_results_dir =  os.path.join(nnunet_data_dir,'results')
script_output_files_dir = get_config()['script_output_files_dir']


import raw
import json

def path_found(dir_or_file):
    if os.path.exists(dir_or_file):
        return {
            'exists': True,
            'reason': ''
            }
    else:
        return {
            'exists': False,
            'reason': f'Not found - {dir_or_file}'
            }

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

def all_plan_conf_dirs_exists(id):
    conf_dirs = plan_conf_dir_list(id)
    for conf_dir in conf_dirs:
        found = path_found(conf_dir)
        if not found['exists']:
            return found
    return {
        'exists':True,
        'reason': ''
    }

def all_processed_images_exist_in_conf_folders(id):
    
    # file_ending
    file_ending = dataset_json(id)['file_ending']
    
    # training image file ids from raw
    raw_train_image_ids = raw.get_training_image_id_list(id)

    # configuration dir list
    conf_dirs = plan_conf_dir_list(id)

    for conf_dir in conf_dirs:
        
        if not os.path.exists(conf_dir):
            return {
                'exists': False,
                'reason': f'configuration folder not found - {conf_dir}'
            }

        # get all files with file_ending
        files = os.listdir(conf_dir)

        # check there are at least two taining images 
        for image_id in raw_train_image_ids:
            matching_files = [f for f in files if image_id in f]
            if len(matching_files) < 2:
                return {
                    'exists':False,
                    'reason':f'Less than 2 files found for image id [{image_id} in the configuration folder: {conf_dir}]'
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

def pp_complated(id):
    s = pp_status(id)
    return s['case_dir_exists']['exists'] and \
            s['nnUNetPlans_json_exists']['exists'] and \
            s['dataset_json_exists']['exists'] and \
            s['all_plan_conf_dirs_exists']['exists'] and \
            s['all_processed_images_exist_in_conf_folders']['exists']


def pp_status(id):

    if not case_dir_exists(id)['exists']:
        return {
            "case_dir_exists": case_dir_exists(id),
            "nnUNetPlans_json_exists": {'exists':False, 'reason':''},
            "dataset_json_exists": {'exists':False, 'reason':''},
            "all_plan_conf_dirs_exists": {'exists':False, 'reason':''},
            "all_processed_images_exist_in_conf_folders": {'exists':False, 'reason':''}
        }
    else:
        return {
            "case_dir_exists": case_dir_exists(id),
            "nnUNetPlans_json_exists": plan_json_exists(id),
            "dataset_json_exists": dataset_json_exists(id),
            "all_plan_conf_dirs_exists": all_plan_conf_dirs_exists(id),
            "all_processed_images_exist_in_conf_folders": all_processed_images_exist_in_conf_folders(id)
        }

def pp_submit_job(id):
    

    job_num = id[7:10]

    job_name = f'pp_{job_num}'

    print(f'checking job {job_name} is if scheduled or already running')
    from simple_slurm_server import slurm_commands
    print(f'pp jost submitted: {job_num}')
    jobs = slurm_commands.get_jobs_of_user('jinkokim')
    
    jobs_of_name = [job for job in jobs if job['name'] == job_name]

    if len(jobs_of_name) > 0:
        print(f'job is already in the queue ')
        job = jobs_of_name[0]
        print(json.dumps(job, indent=4))
        return 
    
    conf = get_config()
    venv_dir = conf['venv_dir']
    nnunet_dir = conf['nnunet_dir']
    raw_dir = conf['raw_dir']
    preprocessed_dir = conf['preprocessed_dir']
    results_dir = conf['results_dir']

    job_name = f'pp_{job_num}'
    partition = 'gpu'
    num_of_nodes = 1
    ntasks_per_node = 28
    num_of_gpus_per_node = 1
    num_of_hours = 1
    email = 'jinkoo.kim@stonybrook.edu'
    dataset_num = job_num
    planner='ExperimentPlanner'

    cmd_line = f'nnUNetv2_plan_and_preprocess -d {dataset_num} -pl {planner} --verbose --verify_dataset_integrity '
    
    case_scripts_dir = os.path.join(script_output_files_dir, job_num)
    if not os.path.exists(case_scripts_dir):
        os.makedirs(case_scripts_dir)

    script_file = os.path.join(case_scripts_dir, 'pp.slurm')

    log_file = script_file+'.log'
    
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

export nnUNet_raw= "{raw_dir}"
export nnUNet_preprocessed= "{preprocessed_dir}"
export nnUNet_results= "{results_dir}"

{cmd_line}
'''
    
    print(f'saving script file - {script_file}')
    with open(script_file, 'w') as file:
        file.write(script)


    cmd = f'module load slurm && sbatch {script_file}'
    print(f'running "{cmd}"')
    import subprocess
    subprocess.run(cmd, shell=True)

    from simple_slurm_server import slurm_commands

    
    print(f'pp jost submitted: {job_num}')

    jobs = slurm_commands.get_jobs_of_user('jinkokim')

    print(f'slurm jobs')
    print(json.dumps(jobs, indent=4))
    




if __name__ == '__main__':
    print('******** dataset ready for pp *********************')
    id_list = raw.get_dataset_id_list_ready_for_pp(min_num_of_training_images=10)

    # get pp status list
    id_list_to_pp = []
    for id in id_list:
        print(f'=== {id} ===')
        if pp_complated(id):
            print(f'\t{id} - Completed')
        else:
            print(f'\t{id} - NOT completed')
            print(json.dumps(pp_status(id), indent=4))
            id_list_to_pp.append(id)

            
    # submit pp jobs
    print('****submititng pp jobs****')
    for id in id_list_to_pp:
        pp_submit_job(id)

    print('done')





