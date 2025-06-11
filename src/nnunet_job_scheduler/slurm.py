from nnunet_job_scheduler.logger import log, log_exception
from simple_slurm_server import slurm_commands
import json, os
import platform
from nnunet_job_scheduler.config import config 

def save_jobs_of_user():
    slurm_user = config['slurm_user']
    slurm_jobs_dir = config['slurm_jobs_dir']

    log(f'slurm_user={slurm_user}')
    jobs = slurm_commands.get_jobs_of_user(slurm_user)
    log(f'Num of jobs for {slurm_user}={len(jobs)}')
    for job in jobs:
        log(json.dumps(job, indent=4))

    slurm_jobs_file = os.path.join(slurm_jobs_dir, f'jobs.{platform.node()}.{slurm_user}.json')
    log(f'saving slurm jobs to file: {slurm_jobs_file}')
    with open(slurm_jobs_file, 'w',encoding='utf-8') as f:
        json.dump(jobs, f, indent=4, ensure_ascii=False)


def save_jobs_of_all_users():
    slurm_jobs_dir = config['slurm_jobs_dir']

    jobs = slurm_commands.get_jobs()
    log(f'Num of jobs ={len(jobs)}')
    for job in jobs:
        log(json.dumps(job, indent=4))

    slurm_jobs_file = os.path.join(slurm_jobs_dir, f'jobs.{platform.node()}.all.json')
    log(f'saving slurm jobs to file: {slurm_jobs_file}')
    with open(slurm_jobs_file, 'w',encoding='utf-8') as f:
        json.dump(jobs, f, indent=4, ensure_ascii=False)