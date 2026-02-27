
import os, json
from pathlib import Path

from nnunet_job_scheduler.logger import log


from nnunet_job_scheduler.utils import path_found

from nnunet_job_scheduler import raw, pp, tr, utils

from nnunet_job_scheduler.config import config

nnunet_predictions_dir =  config['predictions_dir']

slurm_user = config['slurm_user']
slurm_email = config['slurm_email']
slurm_num_of_tasks_per_node = config['slurm_num_of_tasks_per_node']
slurm_num_of_nodes = config['slurm_num_of_nodes']
slurm_num_of_hours = config['slurm_num_of_hours']
slurm_partition = config['slurm_partition']
slurm_num_of_gpus_per_node = config['slurm_num_of_gpus_per_node']

nnunet_trainer = config['nnunet_trainer']

trainer = nnunet_trainer
plans = config['nnunet_plans']
folds = [0,1,2,3,4]

def id_list():
    """Get a list of data set."""
    pattern = r"^Dataset\d{3}_.+$"  # Regex for Datasetxxx_yyyyyy format
    return [entry.name for entry in Path(nnunet_predictions_dir).iterdir() if entry.is_dir() and re.match(pattern, entry.name)]

def case_dir(id):
    return os.path.join(nnunet_predictions_dir, id)

def case_dir_exists(id):
    return path_found(case_dir(id))

def req_dir(id, req_dirname):
    return os.path.join(case_dir(id), req_dirname)

def req_dir_list(id):
    import glob
    dirs = [d for d in glob.glob(f"{case_dir(id)}/req_[0-9][0-9][0-9]") 
        if os.path.isdir(d)]
    return sorted(dirs)

def req_dirname_list(id):
    import glob
    dirs = [os.path.basename(d) for d in req_dir_list(id)]
    return dirs

def req_num_list(id):
    import glob
    nums = [ int(dirname.replace('req_', '')) for dirname in req_dirname_list(id)]
    return nums(sorted)

def input_image_id_list_for_req(id, req_dirname):
    ext = raw.file_ending(id)
    return utils.file_id_list( req_dir(id, req_dirname), ext, len('_0000')+len(ext))

def output_label_id_list_for_req(id, req_dirname):
    ext = raw.file_ending(id)
    output_dir = req_output_dir(id, req_dirname)
    return utils.file_id_list( output_dir , ext, len(ext))

def req_output_dir(id, req_dirname):
    return utils._join_dir(req_dir(id, req_dirname), 'outputs')

import re

def status(id):
    if not case_dir_exists(id)['exists']:
        return {
            "case_dir_exists": case_dir_exists(id),
        }
    else:
        import utils
        return {
            "case_dir_exists": case_dir_exists(id),
        }

def completed(id):
    s = status(id)
    for key in s.keys():
        if 'exists' in s[key] and not s[key]['exists']:
            return False
    return True

def submit_slurm_job(id, req_dirname):
    log(f'submit_slurm_job(id={id}, req_dirname={req_dirname})')
    
    dataset_num = id[7:10]
    req_num = req_dirname.replace('req_', '')

    input_dir = req_dir(id, req_dirname)
    output_dir = req_output_dir(id, req_dirname)

    job_name = f'pr{dataset_num}{req_num}'
    print(f'job_name={job_name}')

    log(f'checking if job {job_name} is already in the queue or running')
    from nnunet_job_scheduler import slurm_commands
    job = slurm_commands.get_job_from_job_name(job_name, slurm_user)
    if job is not None:
        log(f'job is already in the queue ')
        log(json.dumps(job, indent=4))
        return 
    
    conf = config
    venv_dir = conf['venv_dir']
    nnunet_dir = conf['nnunet_dir']
    raw_dir = conf['raw_dir']
    preprocessed_dir = conf['preprocessed_dir']
    results_dir = conf['results_dir']
    configuration = '2d' if raw.is_2d(id) else '3d_lowres'
    plans = conf['nnunet_plans']

    partition = slurm_partition
    num_of_nodes = slurm_num_of_nodes
    ntasks_per_node = slurm_num_of_tasks_per_node
    num_of_gpus_per_node = slurm_num_of_gpus_per_node
    num_of_hours = slurm_num_of_hours
    email = slurm_email

    cmd_line = f'nnUNetv2_predict -d {id} -i {input_dir} -o {output_dir} -f  0 1 2 3 4 -tr {trainer} -c {configuration} -p {plans}'

    log(f'cmd_line={cmd_line}')

    script_output_files_dir = conf['script_output_files_dir']
    case_scripts_dir = os.path.join(script_output_files_dir, dataset_num)
    if not os.path.exists(case_scripts_dir):
        os.makedirs(case_scripts_dir)

    ### script file
    script_file = os.path.join(case_scripts_dir, job_name+'.slurm')

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
    
    #log(f'script={script}')
    
    log(f'saving script file - {script_file}')
    with open(script_file, 'w') as file:
        file.write(script)


    cmd = f'module load slurm && sbatch {script_file}'
    log(f'running "{cmd}"')
    import subprocess
    subprocess.run(cmd, shell=True)

    from nnunet_job_scheduler import slurm_commands

    
    log(f'pr jost submitted: {dataset_num}')

    jobs = slurm_commands.get_jobs_of_user(slurm_user)

    log(f'slurm jobs')
    log(json.dumps(jobs, indent=4))
  
def check_and_submit_pr_jobs():
     log(f'=== check_and_submit_pr_jobs() ===')
     
     for id in id_list():
        if not tr.completed(id):
            log(f'{id} - training NOT completed.')
            continue 
        
        req_dirnames = req_dirname_list(id)
        log(f'req_dirnames={req_dirnames}')

        for req_dirname in req_dirnames:
            log(f'req_dirname={req_dirname}')
            input_image_ids = input_image_id_list_for_req(id, req_dirname)
            log(f'input_image_ids={input_image_ids}')

            output_label_ids = output_label_id_list_for_req(id, req_dirname)
            log(f'output_label_ids={output_label_ids}')
            if set(input_image_ids).issubset(set(output_label_ids)): # check if all input images have corresponding output labels
                log(f'found all predicted outputs in {req_output_dir(id, req_dirname)}')
                continue
   
            log(f'not all inputs are in the outputs...so...')
            log(f'submitting prediction job for {req_dirname} in {id}')
            
            submit_slurm_job(id, req_dirname)


if __name__ == '__main__':
   check_and_submit_pr_jobs()
   log('done')



        

    