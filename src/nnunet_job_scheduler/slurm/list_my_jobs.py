if __name__ == '__main__':
    from nnunet_job_scheduler import slurm_commands
    from nnunet_job_scheduler.config import config
    import json

    slurm_user = config['slurm_user']
    jobs = slurm_commands.get_jobs_of_user(slurm_user)
    print(f'Num of jobs={len(jobs)}')
    for job in jobs:
        print(json.dumps(job, indent=4))

    print('done')
