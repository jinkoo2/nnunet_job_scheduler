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

    def upload_model_chunked(self, job_id: str, zip_path: str, chunk_size_mb: int = 50) -> dict:
        """Upload a model ZIP in chunks to work around ingress body-size limits."""
        chunk_size = chunk_size_mb * 1024 * 1024
        file_size = os.path.getsize(zip_path)
        total_chunks = (file_size + chunk_size - 1) // chunk_size

        log(f'Chunked model upload: {zip_path} ({file_size / 1024 / 1024:.1f} MB, {total_chunks} chunks)')

        # Init
        resp = self._post(f'/api/jobs/{job_id}/model/upload/init', json={
            'total_chunks': total_chunks,
            'total_size': file_size,
        })
        upload_id = resp['upload_id']
        log(f'Model upload session: {upload_id}')

        # Upload chunks
        with open(zip_path, 'rb') as f:
            for i in range(total_chunks):
                data = f.read(chunk_size)
                r = requests.post(
                    f"{self.base}/api/jobs/{job_id}/model/upload/{upload_id}/chunk/{i}",
                    headers=self.headers,
                    data=data,
                    timeout=300,
                )
                r.raise_for_status()
                log(f'  chunk {i + 1}/{total_chunks} uploaded')

        # Complete
        log('Completing model upload...')
        return self._post(f'/api/jobs/{job_id}/model/upload/{upload_id}/complete', timeout=120)
