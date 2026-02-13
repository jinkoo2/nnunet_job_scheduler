if __name__ == '__main__':
    from nnunet_job_scheduler import slurm_commands
    slurm_user = 'jinkokim'
    jobs = slurm_commands.get_jobs_of_user(slurm_user)
    job_ids = [job['jobid'] for job in jobs]

    if len(job_ids) == 0:
        print('No job to cancel')
        exit(0)

    for jobid in job_ids:
        print(f'cancelling job {jobid}')
        slurm_commands.cancel_job(jobid)

    print('done')

