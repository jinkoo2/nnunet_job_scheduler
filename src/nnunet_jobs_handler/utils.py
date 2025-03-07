import os



def _join_dir(dir1, dir2):
    joined = os.path.join(dir1, dir2)
    if not os.path.exists(joined):
        os.makedirs(joined)
    return joined

def _error(msg):
    raise Exception(f'Error: msg')

def _info(msg):
    print(msg)

def _must_exist(path):
    if not os.path.exists(path):
        raise Exception('Paht not found:{path}')

def get_current_datetime_str():    
    from datetime import datetime

    # Get current date and time
    now = datetime.now()

    # Format the datetime as a folder-compatible string
    return now.strftime("%Y%m%d_%H%M%S")

def copy_file_to_folder(source_file, destination_folder):
    import shutil
    import os

    # Get the full destination path (including the file name)
    destination_file = os.path.join(destination_folder, os.path.basename(source_file))

    # Copy the file
    shutil.copy(source_file, destination_file)

import os

def get_unique_file_path(base_filename, directory):
    # Extract the base name and extension
    base_name, extension = os.path.splitext(base_filename)
    
    # Start with the original filename
    file_path = os.path.join(directory, base_filename)
    counter = 1

    # Loop until we find a filename that doesn't exist
    while os.path.exists(file_path):
        # Generate new filename with counter
        file_path = os.path.join(directory, f"{base_name}_{counter}{extension}")
        counter += 1
       
    return file_path



def path_found(dir_or_file):
    if os.path.exists(dir_or_file):
        return {
            'exists': True,
            'reason': ''
            }
    else:
        return {
            'exists': False,
            'reason': f'Not found - {dir_or_file}'
            }



import os
from datetime import datetime

def list_files_with_mtime(directory, extension=None):
    """
    List all files in a directory and its subdirectories, returning a sorted list of dictionaries.
    
    Args:
        directory (str): The path to the directory to scan.
        extension (str, optional): Filter files by extension (e.g., '.txt').
    
    Returns:
        list: A list of dictionaries, sorted by last modified time (newest to oldest).
    """
    file_list = []
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if extension and not filename.endswith(extension):
                continue  # Skip files that don’t match the extension
            
            file_path = os.path.join(root, filename)
            try:
                mtime_timestamp = os.path.getmtime(file_path)
                mtime = datetime.fromtimestamp(mtime_timestamp)
                file_info = {
                    'path': file_path.replace(directory,''),
                    'mtime': mtime
                }
                file_list.append(file_info)
            except (OSError, PermissionError) as e:
                print(f"Could not access {file_path}: {e}")
    
    file_list.sort(key=lambda x: x['mtime'], reverse=True)
    return file_list

# Example usage
if __name__ == "__main__":
    from config import config
    directory_path = os.path.join(config['raw_dir'], 'Dataset847_FourCirclesOnJawCalKv2')
    files = list_files_with_mtime(directory_path)  # Only .py files
    for file in files:  # Combining with your previous question—first 6 files
        print(f"Path: {file['path']}, Last Modified: {file['mtime']}")