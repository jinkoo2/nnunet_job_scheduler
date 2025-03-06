import time, json

from logger import log, log_exception

log('App starting...')

from config import get_config
cfg = get_config()
log(f'=== [config] ===')
for key in cfg:
    log(f'{key}={cfg[key]}')

import pp, tr

while True:

    try:    
        log('=============================================')
        log('====== pp.check_and_submit_pp_jobs() ========')
        log('=============================================')    
        pp.check_and_submit_pp_jobs()
    except Exception as e:
        log('Exception in pp.check_and_submit_pp_jobs()!')
        log_exception(e)

    try:
        log('=============================================')
        log('====== tr.check_and_submit_tr_jobs() ========')
        log('=============================================')    
        tr.check_and_submit_tr_jobs()
    except Exception as e:
        log('Exception in tr.check_and_submit_tr_jobs()!')
        log_exception(e)

    sleep_min = 1
    log(f'sleeping for {sleep_min} minutes...')
    time.sleep(sleep_min * 60)
    
