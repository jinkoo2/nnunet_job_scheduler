
import json
import os
import re
from pathlib import Path

from nnunet_job_scheduler.logger import log, log_exception as LE
from nnunet_job_scheduler import utils 
from nnunet_job_scheduler.config import config

nnunet_data_dir = config['data_dir'] 
nnunet_raw_dir =  config['raw_dir']

def id_list():
    """Get a list of data set."""
    pattern = r"^Dataset\d{3}_.+$"  # Regex for Datasetxxx_yyyyyy format
    return sorted([entry.name for entry in Path(nnunet_raw_dir).iterdir() if entry.is_dir() and re.match(pattern, entry.name)])

def dataset_json_file(id):
    return os.path.join(nnunet_raw_dir, id, 'dataset.json')
def dataset_json_exists(id):
    return path_found(dataset_json_file(id))
def dataset_json(id):
    json_file = dataset_json_file(id)
    try:
        with open(json_file, 'r') as f:
            data = f.read()
            data_dict = json.loads(data)
            data_dict['id'] = id
            return data_dict
    except Exception as e:
        LE(e)
        return None
    
def dataset_json_list():

    dataset_list = [dataset_json(id) for id in id_list()]
    
    # Remove None values (failed reads)
    dataset_list = [ds for ds in dataset_list if ds]

    dataset_list = sorted(dataset_list, key=lambda x: x["id"])

    log(f'dataset_list={dataset_list}')

    return dataset_list


def dataset_num_list():
    num_list = [re.search(r'Dataset(\d+)_', name).group(1) for name in id_list()]
    return num_list

def _file_id_list(dir, ext, n_tail_to_cut_off):
    return utils.file_id_list(dir, ext, n_tail_to_cut_off)

def file_ending(id):

    if not dataset_json_exists(id)['exists']:
        return None

    ds = dataset_json(id)
    return  ds['file_ending'].strip()

def images_tr_file_id_list(id):
    return _file_id_list(images_tr_dir(id), file_ending(id), len('_0000')+len(file_ending(id)))

def labels_tr_file_id_list(id):
    return _file_id_list(labels_tr_dir(id), file_ending(id),len(file_ending(id)))
    
def images_ts_file_id_list(id):
    return _file_id_list(images_ts_dir(id), file_ending(id),len('_0000')+len(file_ending(id)))

def labels_ts_file_id_list(id):
    return _file_id_list(labels_ts_dir(id), file_ending(id),len(file_ending(id)))

def pp_ready(id):
    log(f'pp_ready({id})')
    
    min_num_of_required_training_images = config['min_num_of_required_training_images']
    log(f'min_num_of_required_training_images={min_num_of_required_training_images}')

    # check num of training images
    log(f'checking if the numTraining is > the minimum number of required images.')
    ds = dataset_json(id)
    numTraining = ds['numTraining']
    if numTraining < min_num_of_required_training_images:
        return {'ready':False, 'reason':f'Not enought number of images (N={numTraining}, required={min_num_of_required_training_images}).'}

    # check number of training images in the folder
    num_of_images_found = len(images_tr_file_id_list(id))
    if (num_of_images_found >= numTraining):
        return {'ready':True,'reason':''}
    else:
        file_ending = ds['file_ending']
        return {'ready':False, 'reason':f'Something wrong: Number of training images found (N={num_of_images_found}, file_ending={file_ending}) < numTraining in dataset.json file for {id}'}

from nnunet_job_scheduler.utils import path_found

def case_dir(id):
    return os.path.join(nnunet_raw_dir, id)
def case_dir_exists(id):
    return path_found(case_dir(id))

def is_2d(id):
    if not case_dir_exists(id)['exists']:
        return False 
    if not dataset_json_exists(id)['exists']:
        return False
    
    dataset = dataset_json(id)
    if not 'tensorImageSize' in dataset:
        return False
    return dataset['tensorImageSize'].lower().strip() == '2d'

def is_3d(id):
    if not case_dir_exists(id)['exists']:
        return False 
    if not dataset_json_exists(id)['exists']:
        return False
    
    dataset = dataset_json(id)
    if not 'tensorImageSize' in dataset:
        return False
    return dataset['tensorImageSize'].lower().strip() == '3d'

def images_tr_dir(id):
    return os.path.join(case_dir(id), 'imagesTr')
def images_tr_dir_exists(id):
    return path_found(images_tr_dir(id))

def labels_tr_dir(id):
    return os.path.join(case_dir(id), 'labelsTr')
def labels_tr_dir_exists(id):
    return path_found(labels_tr_dir(id))

def images_ts_dir(id):
    return os.path.join(case_dir(id), 'imagesTs')
def images_ts_dir_exists(id):
    return path_found(images_ts_dir(id))

def labels_ts_dir(id):
    return os.path.join(case_dir(id), 'labelsTs')
def labels_ts_dir_exists(id):
    return path_found(labels_ts_dir(id))

def status(id):
    if not case_dir_exists(id)['exists']:
        return {
            "case_dir_exists": case_dir_exists(id),
            "dataset_json_exists": {'exists':False, 'reason':''},
            "images_tr_dir_exists": {'exists':False, 'reason':''},
            "labels_tr_dir_exists": {'exists':False, 'reason':''},
            "checkpoint_best_exists_for_all_folds": {'exists':False, 'reason':''},
            "checkpoint_final_exists_for_all_folds": {'exists':False, 'reason':''},
            "validation_summary_file_exists_for_all_folds": {'exists':False, 'reason':''}
        }
    else:
        import utils
        return {
            "case_dir_exists": case_dir_exists(id),
            "dataset_json_exists": dataset_json_exists(id, 'dataset.json'),
            "images_tr_dir_exists": images_tr_dir_exists(id),
            "images_tr_file_id_list": images_tr_file_id_list(id),
            "labels_tr_dir_exists": labels_tr_dir_exists(id),
            "labels_tr_file_id_list": labels_tr_file_id_list(id),
            "images_ts_dir_exists": images_ts_dir_exists(id),
            "images_ts_file_id_list": images_ts_file_id_list(id),
            "labels_ts_dir_exists": labels_ts_dir_exists(id),
            "labels_ts_file_id_list": labels_ts_file_id_list(id),
            "dataset_json": dataset_json(id),
            "pp_ready": pp_ready(id),
            "files": utils.list_files(case_dir(id))
        }
    
def dataset_id_list_ready_for_pp():
    list = []
    for id in id_list():
        ret = pp_ready(id)
        if ret['ready']:
            list.append(id)
    
    return list

import json
if __name__ == '__main__':
   
    print('=== datasets ===')
    print(json.dumps(id_list(), indent=4))

    print('=== dataset num list ===')
    print(json.dumps(dataset_num_list(), indent=4))
    
    print('=== dataset json list ===')
    print(json.dumps(dataset_json_list(), indent=4))

    print('=== datasets ready for pp ===')
    datasets_ready_for_pp = dataset_id_list_ready_for_pp()
    for id in datasets_ready_for_pp:
        print(f'{id} - pp ready')

    print(f'=== training_image_id_list for  {datasets_ready_for_pp[0]} ===')
    image_file_ids = images_tr_file_id_list(datasets_ready_for_pp[0])
    for i,file in enumerate(image_file_ids):
        print(f'image[{i}] {file}')


    print(f'=== training_label_id_list for  {datasets_ready_for_pp[0]} ===')
    image_file_ids = labels_tr_file_id_list(datasets_ready_for_pp[0])
    for i,file in enumerate(image_file_ids):
        print(f'image[{i}] {file}')


    print(f'=== test_image_id_list for  {datasets_ready_for_pp[0]} ===')
    image_file_ids = images_ts_file_id_list(datasets_ready_for_pp[0])
    for i,file in enumerate(image_file_ids):
        print(f'image[{i}] {file}')


    print(f'=== test_label_id_list for  {datasets_ready_for_pp[0]} ===')
    image_file_ids = labels_ts_file_id_list(datasets_ready_for_pp[0])
    for i,file in enumerate(image_file_ids):
        print(f'image[{i}] {file}')

    print('done')




