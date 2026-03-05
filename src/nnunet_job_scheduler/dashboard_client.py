import os
import requests

from nnunet_job_scheduler.config import config
from nnunet_job_scheduler.logger import log


class DashboardClient:
    def __init__(self):
        self.base = config.get('dashboard_url', '').rstrip('/')
        self.headers = {'X-Api-Key': config.get('dashboard_api_key', '')}

    def _get(self, path, params=None):
        r = requests.get(
            f"{self.base}{path}", headers=self.headers, params=params, timeout=30
        )
        r.raise_for_status()
        return r.json()

    def _post(self, path, json=None, files=None, timeout=30):
        headers = dict(self.headers)
        if files:
            headers.pop('Content-Type', None)
        r = requests.post(
            f"{self.base}{path}",
            headers=headers,
            json=json,
            files=files,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()

    def _put(self, path, json=None):
        r = requests.put(
            f"{self.base}{path}", headers=self.headers, json=json, timeout=30
        )
        r.raise_for_status()
        return r.json()

    def register_worker(self, name: str) -> dict:
        return self._post('/api/workers/register', json={'name': name})

    def list_datasets(self) -> list:
        return self._get('/api/datasets/')

    def create_job(self, dataset_id: str, worker_id: str, configuration: str) -> dict:
        return self._post('/api/jobs/', json={
            'dataset_id': dataset_id,
            'worker_id': worker_id,
            'configuration': configuration,
        })

    def update_job_status(self, job_id: str, status: str) -> dict:
        return self._put(f'/api/jobs/{job_id}/status', json={'status': status})

    def upload_model(self, job_id: str, zip_path: str) -> dict:
        log(f'Uploading model to dashboard: job_id={job_id}, zip={zip_path}')
        with open(zip_path, 'rb') as f:
            return self._post(
                f'/api/jobs/{job_id}/model',
                files={'zip_file': (os.path.basename(zip_path), f, 'application/zip')},
                timeout=3600,
            )
