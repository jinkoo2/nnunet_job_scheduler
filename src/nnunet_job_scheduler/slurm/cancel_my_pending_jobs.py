if __name__ == '__main__':
    from nnunet_job_scheduler import slurm_commands
    slurm_user = 'jinkokim'
    slurm_commands.cancel_pending_jobs_of_user(slurm_user)
    print('done')