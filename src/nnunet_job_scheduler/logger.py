import logging
import os
from datetime import datetime

from nnunet_job_scheduler.config import config

# log dir
log_dir = config["log_dir"] 
print(f'log_dir={log_dir}')
if not os.path.exists(log_dir):
    print('log_dir not found. So, creating one...')
    os.makedirs(log_dir)

# Define log file name with timestamp
#log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
#print(f'log_file={log_file}')

# Set up logger
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)  # Change to logging.INFO if you don't want debug messages

# Formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# File Handler
# Custom FileHandler that updates filename dynamically
class DynamicFileHandler(logging.FileHandler):
    def __init__(self, log_dir, mode='a', encoding=None, delay=False):
        self.log_dir = log_dir
        # Initialize with a dummy filename (will be updated on first emit)
        super().__init__(os.path.join(log_dir, "dummy.log"), mode, encoding, delay)
    
    def emit(self, record):
        # Update the filename based on current date before emitting
        new_log_file = os.path.join(self.log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
        if self.baseFilename != new_log_file:
            # Close the old file and open a new one
            self.close()
            self.baseFilename = new_log_file
            self.stream = self._open()
        # Proceed with normal emit
        super().emit(record)

file_handler = DynamicFileHandler(log_dir, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def log(msg):
    logger.info(msg)
    
# Function to log an exception
def log_exception(e):
    logger.error("Exception occurred", exc_info=e)

def log_and_raise_exception(e):
    logger.error("Exception occurred", exc_info=e)
    raise e

#from fastapi import Request
#def log_request(request: Request):
#    client_ip = request.client.host if request.client else "Unknown IP"
#    user_agent = request.headers.get("User-Agent", "Unknown User-Agent")
#    logger.info(f"Received request from {client_ip}, User-Agent: {user_agent}")

# Example usage:
if __name__ == "__main__":
    logger.info("Logger initialized")
    try:
        1 / 0  # Intentional error
    except Exception as e:
        log_exception(e)
