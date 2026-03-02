if __name__ == '__main__':
    from nnunet_job_scheduler import slurm_commands
    from nnunet_job_scheduler.config import config

    slurm_user = config['slurm_user']
    slurm_commands.cancel_pending_jobs_of_user(slurm_user)
    print('done')