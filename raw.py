
import json
import os
import re
from pathlib import Path

from logger import logger, log_exception as LE, log_and_raise_exception as LER

from config import get_config
nnunet_data_dir = get_config()["data_dir"] 
#print(f'nnunet_data_dir={nnunet_data_dir}')

nnunet_raw_dir =  os.path.join(nnunet_data_dir,'raw')
nnunet_preprocessed_dir =  os.path.join(nnunet_data_dir,'preprocessed')
nnunet_results_dir =  os.path.join(nnunet_data_dir,'results')

def get_dataset_dirs(folder_path):
    """Get a list of data set."""
    pattern = r"^Dataset\d{3}_.+$"  # Regex for Datasetxxx_yyyyyy format
    return [entry.name for entry in Path(folder_path).iterdir() if entry.is_dir() and re.match(pattern, entry.name)]

def read_dataset_json(dirname):
    """Read dataset.json file asynchronously."""
    json_file = os.path.join(nnunet_raw_dir, dirname, 'dataset.json')
    try:
        with open(json_file, 'r') as f:
            data = f.read()
            data_dict = json.loads(data)
            data_dict['id'] = dirname
            return data_dict
    except Exception as e:
        LE(f'Failed reading file {json_file}. Exception: {e}')
        return None
    
def get_dataset_json_list():
    """Retrieve dataset list asynchronously."""
    dirnames = get_dataset_dirs(nnunet_raw_dir)
    dataset_list = [read_dataset_json(dirname) for dirname in dirnames]
    
    # Remove None values (failed reads)
    dataset_list = [ds for ds in dataset_list if ds]

    dataset_list = sorted(dataset_list, key=lambda x: x["id"])

    logger.info(f'dataset_list={dataset_list}')

    return dataset_list

def get_dataset_id_list():
    return sorted(get_dataset_dirs(nnunet_raw_dir))

def get_dataset_num_list():
    id_list = get_dataset_id_list()
    num_list = [re.search(r'Dataset(\d+)_', name).group(1) for name in id_list]
    return num_list

def get_num_of_training_images(id, file_ending):
    dir = os.path.join(nnunet_raw_dir, id, 'imagesTr')
    if not os.path.exists(dir):
        return -1
    
    return len([f for f in os.listdir(dir) if f.endswith(file_ending)])

def get_training_image_id_list(id):
    
    # file_ending
    dataset_json = read_dataset_json(id)
    #print(json.dumps(dataset_json, indent=4))

    file_ending = dataset_json['file_ending']

    # get unique image ids
    dir = os.path.join(nnunet_raw_dir, id, 'imagesTr')
    
    image_files = [f for f in os.listdir(dir) if f.endswith(file_ending)]

    # remove file_ending and _0000
    n_tail = len('_0000')+len(file_ending)
    image_files = [f[0:-n_tail] for f in image_files]
    
    return sorted(list(set(image_files)))

def pp_ready(id, min_num_of_training_images):
    dataset_json = read_dataset_json(id)
    numTraining = dataset_json['numTraining']
    if numTraining < min_num_of_training_images:
        return {'ready':False, 'reason':f'Not enought number of images (N={numTraining}, required={min_num_of_training_images}).'}

    file_ending = dataset_json['file_ending']
    num_of_images_found = get_num_of_training_images(id, file_ending)
    
    if (numTraining > 10 and num_of_images_found >= numTraining):
        return {'ready':True,'reason':''}
    else:
        return {'ready':False, 'reason':f'Something wrong: Number of training images found (N={num_of_images_found}, file_ending={file_ending}) < numTraining in dataset.json file for {id}'}

def get_dataset_id_list_ready_for_pp(min_num_of_training_images):
    list = []
    for id in get_dataset_id_list():
        ret = pp_ready(id, min_num_of_training_images)
        if ret['ready']:
            list.append(id)
    
    return list

import json
if __name__ == '__main__':

    image_file_ids = get_training_image_id_list('Dataset009_Spleen')
    for file in image_file_ids:
        print(file)

    exit(0)
    
    print('=== datasets ===')
    print(json.dumps(get_dataset_id_list(), indent=4))

    print('=== dataset num list ===')
    print(json.dumps(get_dataset_num_list(), indent=4))
    
    print('=== dataset json list ===')
    print(json.dumps(get_dataset_json_list(), indent=4))

    print('Training Ready?')
    for id in get_dataset_id_list():
        ret = pp_ready(id, min_num_of_training_images=10)
        if ret['ready']:
            print(f'dataset[{id}] - Yes')
        else:
            print(f'dataset[{id}] - No ({ret["reason"]})')
    
    datasets_ready_for_training = get_dataset_id_list_ready_for_pp(min_num_of_training_images=10)
    print('')
    print('[Preprocesisng Read]')
    for id in datasets_ready_for_training:
        print(id)

    print('done')




