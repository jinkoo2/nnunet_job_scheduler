import time
from nnunet_job_scheduler.logger import log, log_exception
from nnunet_job_scheduler.config import config
from nnunet_job_scheduler import pp, tr, pr, slurm

def main():

    log('App starting...')

    log(f'=== [config] ===')
    for key in config:
        log(f'{key}={config[key]}')

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

        try:
            log('=============================================')
            log('====== pr.check_and_submit_pr_jobs() ========')
            log('=============================================')
            pr.check_and_submit_pr_jobs()
        except Exception as e:
            log('Exception in pr.check_and_submit_pr_jobs()!')
            log_exception(e)

        try:
            log('=============================================')
            log('====== list of slurm jobs====================')
            log('=============================================')
            
            slurm.save_jobs_of_user()
            #slurm.save_jobs_of_all_users()
        except Exception as e:
            log('Exception in slurm job saving!')
            log_exception(e)

        sleep_min = 60
        log(f'sleeping for {sleep_min} minutes...')
        time.sleep(sleep_min * 60)

if __name__ == "__main__":
    main()
