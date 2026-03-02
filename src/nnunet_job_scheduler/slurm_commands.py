import subprocess
import json

def run_command(command):
    """Run a shell command with 'module load slurm'."""
    try:
        full_command = f"module load slurm && {command}"
        result = subprocess.run(
            ["bash", "-c", full_command], stdout=subprocess.PIPE, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise

# Format: jobid(18) partition(9) name(200) user(8) state(2) time(12) nodes(6) nodelist(rest)
# %.200j gives full job name (default squeue truncates to 8 chars)
SQUEUE_FORMAT = '%.18i %.9P %.200j %.8u %.2t %.12M %.6D %R'

def get_jobs():
    command = f"squeue -o '{SQUEUE_FORMAT}'"
    result = run_command(command)
    return parse_squeue_results(result)

def parse_squeue_results(result):
    lines = result.strip().split('\n')
    if not lines:
        return []
    headers = ['jobid', 'partition', 'name', 'user', 'st', 'time', 'nodes', 'nodelist']
    jobs = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        job_dict = {
            headers[0]: parts[0],
            headers[1]: parts[1],
            headers[2]: parts[2],
            headers[3]: parts[3],
            headers[4]: parts[4],
            headers[5]: parts[5],
            headers[6]: parts[6],
            headers[7]: ' '.join(parts[7:]) if len(parts) > 7 else '',
        }
        jobs.append(job_dict)
    return jobs

def get_jobs_of_user(user_id):
    command = f"squeue -u {user_id} -o '{SQUEUE_FORMAT}'"
    result = run_command(command)
    return parse_squeue_results(result)

def get_job_from_job_name(job_name, user_id=None):
    if user_id is None:
        jobs = get_jobs()
    else:
        jobs = get_jobs_of_user(user_id)
    jobs_found = [job for job in jobs if job['name'] == job_name]
    if len(jobs_found) == 0:
        return None
    elif len(jobs_found) == 1:
        return jobs_found[0]
    else:
        raise Exception(f"More than 1 job found with job_name={job_name}. jobs_found={jobs_found}")

def get_job(job_id):
    command = "scontrol show job " + job_id
    output = run_command(command)
    job_details = {}
    for line in output.split("\n"):
        for item in line.split():
            if "=" in item:
                key, value = item.split("=", 1)
                job_details[key] = value
    return job_details

def cancel_job(job_id: str):
    command = f"scancel {job_id}"
    run_command(command)

def cancel_jobs_of_user(user_id: str):
    command = f"scancel -u {user_id}"
    run_command(command)

def cancel_pending_jobs_of_user(user_id: str):
    command = f"scancel -u {user_id} -t PENDING"
    run_command(command)


def suspend_job(job_id: str):
    command = f"scontrol suspend {job_id}"
    run_command(command)

def resume_job(job_id: str):
    command = f"scontrol resume {job_id}"
    run_command(command)
