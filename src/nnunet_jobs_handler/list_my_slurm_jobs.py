if __name__ == '__main__':
    from simple_slurm_server import slurm_commands
    import json
    slurm_user = 'jinkokim'
    jobs = slurm_commands.get_jobs_of_user(slurm_user)
    print(f'Num of jobs={len(jobs)}')
    for job in jobs:
        print(json.dumps(job, indent=4))

    print('done')
