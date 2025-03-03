
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
    from simple_slurm_server import slurm_commands

    jobs = slurm_commands.get_jobs()

    print(f'slurm jobs')
    print(json.dumps(jobs, indent=4))
    
    
    print(f'pp jost submitted: {id}')


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





