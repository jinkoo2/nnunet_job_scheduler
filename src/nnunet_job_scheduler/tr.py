
import os, json
from pathlib import Path

from nnunet_job_scheduler.logger import log
from nnunet_job_scheduler.utils import path_found
from nnunet_job_scheduler import raw, pp, utils

from nnunet_job_scheduler.config import get_config

from nnunet_job_scheduler.config import config
nnunet_results_dir =  config['results_dir']
slurm_user = config['slurm_user']
slurm_email = config['slurm_email']

slurm_num_of_tasks_per_node = config['slurm_num_of_tasks_per_node']
slurm_num_of_nodes = config['slurm_num_of_nodes']
slurm_num_of_hours = config['slurm_num_of_hours']
slurm_partition = config['slurm_partition']
slurm_num_of_gpus_per_node = config['slurm_num_of_gpus_per_node']

nnunet_trainer = config['nnunet_trainer']

trainer = nnunet_trainer
plans = 'nnUNetPlans'
folds = [0,1,2,3,4]


def id_list():
    """Get a list of data set."""
    pattern = r"^Dataset\d{3}_.+$"  # Regex for Datasetxxx_yyyyyy format
    return [entry.name for entry in Path(nnunet_results_dir).iterdir() if entry.is_dir() and re.match(pattern, entry.name)]

def case_dir(id):
    return os.path.join(nnunet_results_dir, id)
def case_dir_exists(id):
    return path_found(case_dir(id))

def conf_2d_dir(id):
    configuration = '2d'
    conf_dirname = f'{trainer}__{plans}__{configuration}'
    return os.path.join(nnunet_results_dir, id, conf_dirname)
def conf_2d_dir_exists(id):
    return path_found(conf_2d_dir(id))

def conf_3d_lowres_dir(id):
    configuration = '3d_lowres'
    conf_dirname = f'{trainer}__{plans}__{configuration}'
    return os.path.join(nnunet_results_dir, id, conf_dirname)
def conf_3d_lowres_dir_exists(id):
    return path_found(conf_3d_lowres_dir(id))

def conf_3d_highres_dir(id):
    configuration = '3d_highres'
    conf_dirname = f'{trainer}__{plans}__{configuration}'
    return os.path.join(nnunet_results_dir, id, conf_dirname)
def conf_3d_highres_dir_exists(id):
    return path_found(conf_3d_highres_dir(id))

def conf_3d_fullres_dir(id):
    configuration = '3d_fullres'
    conf_dirname = f'{trainer}__{plans}__{configuration}'
    return os.path.join(nnunet_results_dir, id, conf_dirname)

def conf_dir_for_config(id, configuration):
    """Return results dir for a given configuration (2d, 3d_lowres, 3d_highres, 3d_fullres)."""
    if configuration == '2d':
        return conf_2d_dir(id)
    if configuration == '3d_lowres':
        return conf_3d_lowres_dir(id)
    if configuration == '3d_highres':
        return conf_3d_highres_dir(id)
    if configuration == '3d_fullres':
        return conf_3d_fullres_dir(id)
    raise ValueError(f'Unknown configuration: {configuration}')

def conf_dir(id):
    # check if 2d
    if raw.is_2d(id):
        return conf_2d_dir(id)
    else:
        return conf_3d_lowres_dir(id)

def conf_dir_exists(id):
    return path_found(conf_dir(id))

def fold_dir(id, fold):
    return os.path.join(conf_dir(id), f'fold_{fold}')

def fold_dir_exists(id, fold):
    return path_found(fold_dir(id, fold))

def cache_dir(id, fold):
    return utils._join_dir(fold_dir(id, fold), '__cache__')

def training_log_files_for_fold(id, fold):
    if not os.path.exists(fold_dir(id, fold)):
        return []

    files = utils.list_files(fold_dir(id,fold), include_sub_folders=False, extension='.txt')
    files = [f for f in files if os.path.basename(f['path']).startswith('training_log_')]
    return files

def training_log_files(id):
    ret = {}
    for fold in folds:
        ret[f'fold_{fold}'] = training_log_files_for_fold(id, fold)
    return ret

def training_log_for_fold(id, fold):

    if not fold_dir_exists(id, fold)['exists']:
        return ''
    
    files = training_log_files_for_fold(id, fold)
    if len(files) == 0:
        return ''
    
    # check if there is a log file already
    latest_log_file_time = files[-1]['mtime_sec_since_1970utc']
    log_file = os.path.join(cache_dir(id, fold), f'training_log_{latest_log_file_time}.txt' )
    if os.path.exists(log_file):
        log(f'A cached log_file found. Rreturning the file. file={log_file}')
        with open(log_file, 'r') as f:
            return f.read()
        
    filenames = [f['path'] for f in files]
    dir = fold_dir(id, fold)

    full_path_list = [os.path.join(dir, filename) for filename in filenames]

    log_text = ''
    for full_path in full_path_list:
        with open(full_path, 'r') as f:
            txt = f.read()
            log_text = log_text + txt + '\n'

    # save as cache
    log('removing all training_log files in the cache folder')
    existing_log_filenames = [f for f in os.listdir(cache_dir(id,fold)) if f.startswith('training_log_') and f.endswith('.txt')]
    for file in existing_log_filenames:
        full_path = os.path.join(cache_dir(id, fold), file)
        os.remove(full_path)
        log(f"Deleted: {full_path}")
    
    log(f'saving training_log as cache to {log_file}')
    with open(log_file, 'w') as f:
        f.write(log_text)

    return log_text

def training_logs(id):
    logs = {}
    for fold in folds:
        logs[f'fold_{fold}'] = training_log_files_for_fold(id, fold)
    return logs

import re
from ast import literal_eval

def parse_epoch_data_from_training_log(log_text):
    """
    Parse a training log and extract a list of dictionaries with epoch data.
    
    Args:
        log_text (str): The full text of the training log.
    
    Returns:
        list: List of dictionaries with keys 'Epoch', 'Current learning rate',
              'train_loss', 'val_loss', and 'Pseudo dice'.
    """
    # Split the log into lines
    lines = log_text.strip().split('\n')
    
    # List to store epoch dictionaries
    epoch_data = []
    current_epoch = {}
    
    # Regex patterns for extracting values
    patterns = {
        'Epoch': r'Epoch (\d+)',
        'Current learning rate': r'Current learning rate: ([\d.e-]+)',
        'train_loss': r'train_loss ([\d.-]+)',
        'val_loss': r'val_loss ([\d.-]+)',
        'Pseudo dice': r'Pseudo dice (\[[\d., ]+\])'
    }
    
    for line in lines:
        # Check if line is a separator (blank or just timestamp)
        if not line.strip() or re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}:\s*$', line):
            if current_epoch:  # If we have data for an epoch, add it to the list
                # Only add if all required keys are present
                if all(key in current_epoch for key in patterns.keys()):
                    epoch_data.append(current_epoch)
                current_epoch = {}  # Reset for next epoch
            continue
        
        # Try to match each pattern
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                value = match.group(1)
                if key == 'Pseudo dice':
                    # Convert string representation of list to actual list
                    current_epoch[key] = literal_eval(value)
                elif key in ['train_loss', 'val_loss', 'Current learning rate']:
                    # Convert to float
                    current_epoch[key] = float(value)
                else:
                    # Epoch number as int
                    current_epoch[key] = int(value)
    
    # Add the last epoch if it has all required keys
    if current_epoch and all(key in current_epoch for key in patterns.keys()):
        epoch_data.append(current_epoch)
    
    return epoch_data

def training_epoch_data_for_fold(id, fold):

    # check the cache
    log_files = training_log_files_for_fold(id, fold)
    if len(log_files) ==0:
        return []

    latest_log_file_time = log_files[-1]['mtime_sec_since_1970utc']
    epoch_data_file = os.path.join(cache_dir(id, fold), f'epoch_data_{latest_log_file_time}.json')
    if os.path.exists(epoch_data_file):
        log(f'cached training epoch data file found. returning it. file={epoch_data_file}')
        with open(epoch_data_file, 'r') as f:
            return json.load(f)

    epoch_data = parse_epoch_data_from_training_log(training_log_for_fold(id, fold))

    # remove previous cache files
    log('removing all epoch data files in the cache folder')
    existing_filenames = [f for f in os.listdir(cache_dir(id,fold)) if f.startswith('epoch_data_') and f.endswith('.json')]
    for file in existing_filenames:
        full_path = os.path.join(cache_dir(id, fold), file)
        os.remove(full_path)
        log(f"Deleted: {full_path}")

    # save as cache
    log(f'saving epoch data as cache to {epoch_data_file}')
    with open(epoch_data_file, 'w') as f:
        json.dump(epoch_data, f, indent=4)

    return epoch_data

def training_epoch_data(id):
    data = {}
    for fold in folds:
        data[f'fold_{fold}'] = training_epoch_data_for_fold(id, fold)
    return data

def all_fold_dirs_exists(id):
    if conf_dir_exists(id)['exists']:
        missing_folds = []
        for fold in folds:
            found = path_found(fold_dir(id, fold))
            if not found['exists']:
                missing_folds.append(fold)
        if len(missing_folds) == 0:
            return {'exists': True}
        else:
            return {'exists': False, 'reason':f'output dirs missing for folds = {missing_folds})', 'folds': missing_folds }
    else:
        return {'exists': False, 'reason':f'conf_dir not fond:{conf_dir(id)}'}

def exists_in_conf_dir(id, filename):
    if conf_dir_exists(id)['exists']:
        return path_found(os.path.join(conf_dir(id), filename))
    else:
        return {'exists': False, 'reason':f'conf_dir not fond:{conf_dir(id)}'}

def exists_in_fold_dir(id, fold, filename):
    if fold_dir_exists(id, fold)['exists']:
        return path_found(os.path.join(fold_dir(id, fold), filename))
    else:
        return {'exists': False, 'reason':f'conf_dir not fond:{conf_dir(id)}'}

def fold_dir_for_config(id, fold, configuration):
    return os.path.join(conf_dir_for_config(id, configuration), f'fold_{fold}')

def exists_in_fold_dir_for_config(id, fold, configuration, filename):
    fold_path = fold_dir_for_config(id, fold, configuration)
    return path_found(os.path.join(fold_path, filename))

def checkpoint_best_exists(id, fold):
    return exists_in_fold_dir(id, fold, 'checkpoint_best.pth')

def checkpoint_best_exists_for_config(id, fold, configuration):
    """Checkpoint for a specific configuration (use when submitting multi-config jobs)."""
    return exists_in_fold_dir_for_config(id, fold, configuration, 'checkpoint_best.pth')

def checkpoint_best_exists_for_all_folds(id):
    missing_folds = []
    for fold in folds:
        found = checkpoint_best_exists(id, fold)
        if not found['exists']:
            missing_folds.append(fold)
    if len(missing_folds) == 0:
        return {'exists': True}
    else:
        return {'exists': False, 'reason':f'checkpoint_best.json files not found for folds ({missing_folds})', 'folds': missing_folds}

def checkpoint_final_exists(id, fold):
    return exists_in_fold_dir(id, fold, 'checkpoint_final.pth')

def checkpoint_final_exists_for_all_folds(id):
    missing_folds = []
    for fold in folds:
        found = checkpoint_final_exists(id, fold)
        if not found['exists']:
            missing_folds.append(fold)
    if len(missing_folds) == 0:
        return {'exists': True}
    else:
        return {'exists': False, 'reason':f'checkpoint_best.json files not found for folds ({missing_folds})', 'folds': missing_folds}

def validation_dir(id, fold):
    return os.path.join(conf_dir(id), f'fold_{fold}', 'validation')

def validation_dir_exists(id, fold):
    return path_found(validation_dir(id, fold))

def validation_summary_file(id, fold):
    return os.path.join(validation_dir(id, fold), 'summary.json')

def validation_summary_file_exists(id, fold):
    return path_found(validation_summary_file(id, fold))

def validation_summary_file_exists_for_all_folds(id):
    missing_folds = []
    for fold in folds:
        found = validation_summary_file_exists(id, fold)
        if not found['exists']:
            missing_folds.append(fold)
    if len(missing_folds) == 0:
        return {'exists': True}
    else:
        return {'exists': False, 'reason':f'checkpoint_best.json files not found for folds ({missing_folds})', 'folds': missing_folds}
    
def status(id):
    if not case_dir_exists(id)['exists']:
        return {
            "case_dir_exists": case_dir_exists(id),
            "conf_dir_exists": {'exists':False, 'reason':''},
            "plans_json_exists": {'exists':False, 'reason':''},
            "dataset_json_exists": {'exists':False, 'reason':''},
            "dataset_fingerprint_json_exists": {'exists':False, 'reason':''},
            "all_fold_dirs_exists": {'exists':False, 'reason':''},
            "checkpoint_best_exists_for_all_folds": {'exists':False, 'reason':''},
            "checkpoint_final_exists_for_all_folds": {'exists':False, 'reason':''},
            "validation_summary_file_exists_for_all_folds": {'exists':False, 'reason':''}
        }
    else:
        return {
            "case_dir_exists": case_dir_exists(id),
            "conf_dir_exists": conf_dir_exists(id),
            "plans_json_exists": exists_in_conf_dir(id, 'plans.json'),
            "dataset_json_exists": exists_in_conf_dir(id, 'dataset.json'),
            "dataset_fingerprint_json_exists": exists_in_conf_dir(id, 'dataset_fingerprint.json'),
            "all_fold_dirs_exists": all_fold_dirs_exists(id),
            "checkpoint_best_exists_for_all_folds": checkpoint_best_exists_for_all_folds(id),
            "checkpoint_final_exists_for_all_folds": checkpoint_final_exists_for_all_folds(id),
            "validation_summary_file_exists_for_all_folds": validation_summary_file_exists_for_all_folds(id),
            #"files": utils.list_files(case_dir(id)),
            "training_log_files": training_log_files(id),
            #"training_logs": training_logs(id),
            "training_epoch_data": training_epoch_data(id)
        }

def complated(id):
    s = status(id)
    for key in s.keys():
        if 'exists' in s[key] and not s[key]['exists']:
            return False
    return True

def missing_folds_from_status(s):

    if not s['case_dir_exists']['exists']:
        folds = [0,1,2,3,4]
    else:        
        folds = []
        for key in s.keys():
            if 'folds' in s[key]:
                folds.extend(s[key]['folds'])
    print('================')
    log(f'missing_folds_from_status(s) returns folds = {folds}')
    print('================')

    return list(set(folds))

def submit_slurm_job(id, fold, configuration, cont):
    log(f'submit_slurm_job(id={id}, fold={fold}, configuration={configuration}, cont={cont}):')
    
    dataset_num = id[7:10]

    job_name = f'tr_{dataset_num}_{configuration}_{fold}'

    log(f'checking if job {job_name} is already in the queue or running')
    from nnunet_job_scheduler import slurm_commands
    jobs = slurm_commands.get_jobs_of_user(slurm_user)
    log('=== current jobs in the queue ===')
    for i,job in enumerate(jobs):
        log(f"Job[{i}]:{job['jobid']}:{job['name']}:{job['st']}")
        
    jobs_of_name = [job for job in jobs if job['name'] == job_name]

    if len(jobs_of_name) > 0:
        log(f'job is already in the queue ')
        job = jobs_of_name[0]
        log(json.dumps(job, indent=4))
        return 
    
    
    conf = get_config()
    venv_dir = conf['venv_dir']
    nnunet_dir = conf['nnunet_dir']
    raw_dir = conf['raw_dir']
    preprocessed_dir = conf['preprocessed_dir']
    results_dir = conf['results_dir']

    partition = slurm_partition
    num_of_nodes = slurm_num_of_nodes
    ntasks_per_node = slurm_num_of_tasks_per_node
    num_of_gpus_per_node = slurm_num_of_gpus_per_node
    num_of_hours = slurm_num_of_hours
    email = slurm_email

    cmd_line = f'nnUNetv2_train {dataset_num} {configuration} {fold} '
    if cont:
        cmd_line = cmd_line + ' --c'

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
    
    log(f'script={script}')
    
    log(f'saving script file - {script_file}')
    with open(script_file, 'w') as file:
        file.write(script)


    cmd = f'module load slurm && sbatch {script_file}'
    log(f'running "{cmd}"')
    import subprocess
    subprocess.run(cmd, shell=True)

    from nnunet_job_scheduler import slurm_commands

    
    log(f'pp jost submitted: {dataset_num}')

    jobs = slurm_commands.get_jobs_of_user(slurm_user)

    log(f'slurm jobs')
    log(json.dumps(jobs, indent=4))
  
def check_and_submit_tr_jobs():
    log('******** list of pp-complated datasets **************')
    pp_complated_id_list = pp.get_complated_dataset_id_list()
    log(f'pp-complated id_list={pp_complated_id_list}')

    # get tr status list
    tr_cases = []
    for id in pp_complated_id_list:
        log(f'=== {id} ===')
        if complated(id):
            log(f'\t{id} - Completed')
        else:
            log(f'\t{id} - NOT completed')
            s = status(id)
            log(json.dumps(s, indent=4))
            log(f'missing folds={missing_folds_from_status(s)}')
            
            tr_cases.append({'id':id,'folds': missing_folds_from_status(s), 'status': s})
    

    if len(tr_cases) > 0:
        # submit tr jobs
        log(f'****submititng tr jobs [N={len(tr_cases)}]****')
        
        log(f'Tr jobs to submit:{tr_cases}')

        for case in tr_cases:
            id = case['id']
            
            ### configuration
            if raw.is_2d(id):
                configurations = ['2d']
            else:
                configurations = ['3d_lowres', '3d_highres', '3d_fullres']

            for configuration in configurations:
                for fold in case['folds']:
                    ### continue if there is checkpoint_best.pth for this (id, fold, configuration)
                    cont = checkpoint_best_exists_for_config(id, fold, configuration)['exists']

                    log(f'submitting tr slurm job for {id}, fold:{fold}, configuration={configuration}, continue={cont}')
                    submit_slurm_job(id, fold, configuration, cont)
                

if __name__ == '__main__':
    
    
    #check_and_submit_tr_jobs()
    #logs = training_log_files('Dataset105_CBCTBladderRectumBowel')
    #print(json.dumps(logs, indent=4))

    epochs = training_epoch_data_for_fold('Dataset105_CBCTBladderRectumBowel', 0)
    #print(json.dumps(epochs, indent=4))


    exit(0) 
    for id in id_list():
        for fold in folds:
            print(f'{id}, fold_{fold}')
            log_text = training_log_for_fold(id, fold)
            #print(log_text)

            if log_text != '':
                epoch_data = parse_epoch_data_from_training_log(log_text)
                print(epoch_data[-1])
   
    print('done')



        

    