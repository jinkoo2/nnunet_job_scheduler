
import os, json

from logger import log


from utils import path_found

import pp

from config import config
nnunet_results_dir =  config['results_dir']
slurm_user = config['slurm_user']
slurm_email = config['slurm_email']

slurm_num_of_tasks_per_node = config['slurm_num_of_tasks_per_node']
slurm_num_of_nodes = config['slurm_num_of_nodes']
slurm_num_of_hours = config['slurm_num_of_hours']
slurm_partition_for_tr = config['slurm_partition_for_tr']
slurm_num_of_gpus_per_node = config['slurm_num_of_gpus_per_node']

nnunet_trainer = config['nnunet_trainer']

trainer = nnunet_trainer
plans = 'nnUNetPlans'
folds = [0,1,2,3,4]

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

def conf_dir(id):
    # check if 2d
    import pp
    if pp.is_2d(id):
        return conf_2d_dir(id)
    else:
        return conf_3d_lowres_dir(id)

def conf_dir_exists(id):
    return path_found(conf_dir(id))

def fold_dir(id, fold):
    return os.path.join(conf_dir(id), f'fold_{fold}')

def fold_dir_exists(id, fold):
    return path_found(fold_dir(id, fold))

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

def checkpoint_best_exists(id, fold):
    return exists_in_fold_dir(id, fold, 'checkpoint_best.pth')

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
        import utils
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
            "files": utils.list_files_with_mtime(case_dir(id))
        }

def complated(id):
    s = status(id)
    for key in s.keys():
        if not s[key]['exists']:
            return False
    return True

def missing_folds_from_status(s):
    folds = []
    for key in s.keys():
        if 'folds' in s[key]:
            folds.extend(s[key]['folds'])
    return list(set(folds))

def submit_slurm_job(id, fold, configuration, cont):
    log(f'submit_slurm_job(id={id}, fold={fold}, configuration={configuration}, cont={cont}):')
    
    dataset_num = id[7:10]

    job_name = f'tr_{dataset_num}_{fold}'

    log(f'checking if job {job_name} is already in the queue or running')
    from simple_slurm_server import slurm_commands
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
    
    from config import get_config
    conf = get_config()
    venv_dir = conf['venv_dir']
    nnunet_dir = conf['nnunet_dir']
    raw_dir = conf['raw_dir']
    preprocessed_dir = conf['preprocessed_dir']
    results_dir = conf['results_dir']

    partition = slurm_partition_for_tr
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

    script_file = os.path.join(case_scripts_dir, job_name+'.slurm')

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

    
    log(f'pp jost submitted: {dataset_num}')

    jobs = slurm_commands.get_jobs_of_user(slurm_user)

    log(f'slurm jobs')
    log(json.dumps(jobs, indent=4))
  
def check_and_submit_tr_jobs():
    log('******** list of pp-complated datasets **************')
    id_list = pp.get_complated_dataset_id_list()

    # get tr status list
    tr_cases = []
    for id in id_list:
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
        log('****submititng tr jobs****')
        for case in tr_cases:
            id = case['id']
            
            ### configuration
            if pp.is_2d(id):
                configuration = '2d'
            else:
                configuration = '3d_lowres'

            for fold in case['folds']:
                ### continue if there is checkpoint_best.json
                cont = checkpoint_best_exists(id, fold)['exists']

                log(f'submitting tr slurm job for {id}, fold:{fold}, cofiguration={configuration}, continue={cont}')
                submit_slurm_job(id, fold, configuration, cont)
                


if __name__ == '__main__':
    check_and_submit_tr_jobs()
    log('done')

        

    