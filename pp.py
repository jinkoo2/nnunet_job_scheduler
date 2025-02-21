
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



def plan_json_file(id):
    return os.path.join(nnunet_preprocessed_dir, id, 'nnUNetPlans.json')

def plan_json_exists(id):
    if not os.path.exists(plan_json_file(id)):
        #print(f'plan json file not found: {plan_json_file(id)}')
        return False
    return True    

def pp_complated(id):
    if not plan_json_exists(id):
        print('\tplan json file not found')
        return False
    
    return True

if __name__ == '__main__':
    print('*****************************')
    id_list = raw.get_dataset_id_list()

    for id in id_list:
        print(f'=== {id} ===')
        if not pp_complated(id):
            print('\tpp not competed!')

    print('done')





