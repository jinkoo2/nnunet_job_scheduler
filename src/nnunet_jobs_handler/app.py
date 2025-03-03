import time

from config import get_config
cfg = get_config()

while True:
    print('working!')
    print(f"raw_dir={cfg['raw_dir']}")
    print('sleeping...')
    time.sleep(5)
    