import os



def _join_dir(dir1, dir2, makedir=True):
    joined = os.path.join(dir1, dir2)
    if makedir and not os.path.exists(joined):
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


def paths_found(dir_or_file_list):
    for dir_or_file in dir_or_file_list:
        found = path_found(dir_or_file)
        if not found['exists']:
            return found
    return {'exists':True}

import os
from datetime import datetime

def list_files(directory, include_sub_folders=True, extension=None, sort_newer_to_older=False):
    """
    List all files in a directory (and optionally its subdirectories), returning a sorted list of dictionaries.
    Each dictionary contains the relative file path (w.r.t. the input directory), last modified time,
    and file size in bytes.
    
    Args:
        directory (str): The path to the directory to scan.
        include_sub_folders (bool, optional): If True, include files from subdirectories; if False, only the top-level directory.
        extension (str, optional): Filter files by extension (e.g., '.txt').
    
    Returns:
        list: A list of dictionaries, sorted by last modified time (newest to oldest).
              Each dict has 'path' (relative str), 'mtime' (datetime), and 'size' (int) keys.
    """
    file_list = []
    
    # Ensure directory is an absolute path for consistency
    directory = os.path.abspath(directory)
    
    if include_sub_folders:
        # Recursively walk through directory and subdirectories
        for root, _, files in os.walk(directory):
            for filename in files:
                if extension and not filename.endswith(extension):
                    continue  # Skip files that don’t match the extension
                
                # Construct full file path
                full_path = os.path.join(root, filename)
                
                # Get relative path w.r.t. the input directory
                relative_path = os.path.relpath(full_path, directory)
                
                try:
                    # Get last modified time
                    mtime_timestamp = os.path.getmtime(full_path)
                    mtime = datetime.fromtimestamp(mtime_timestamp)
                    
                    # Get file size in bytes
                    size = os.path.getsize(full_path)
                    
                    # Create dictionary with file info
                    file_info = {
                        'path': relative_path,
                        'mtime': mtime.strftime('%Y-%m-%d %H:%M:%S %Z'),  # Include timezone name
                        'mtime_sec_since_1970utc': mtime_timestamp,
                        'size': size
                    }
                    file_list.append(file_info)
                except (OSError, PermissionError) as e:
                    print(f"Could not access {full_path}: {e}")
    else:
        # Only process the top-level directory
        for filename in os.listdir(directory):
            full_path = os.path.join(directory, filename)
            
            # Skip if it’s not a file or doesn’t match the extension
            if not os.path.isfile(full_path) or (extension and not filename.endswith(extension)):
                continue
            
            # Get relative path (will just be the filename since it’s top-level)
            relative_path = os.path.relpath(full_path, directory)
            
            try:
                # Get last modified time
                mtime_timestamp = os.path.getmtime(full_path)
                mtime = datetime.fromtimestamp(mtime_timestamp)
                
                # Get file size in bytes
                size = os.path.getsize(full_path)
                
                # Create dictionary with file info
                file_info = {
                    'path': relative_path,
                    'mtime': mtime.strftime('%Y-%m-%d %H:%M:%S %Z'),  # Include timezone name
                    'mtime_sec_since_1970utc': mtime_timestamp,
                    'size': size
                }
                file_list.append(file_info)
            except (OSError, PermissionError) as e:
                print(f"Could not access {full_path}: {e}")
    
    # Sort by mtime
    if sort_newer_to_older:
        file_list.sort(key=lambda x: x['mtime_sec_since_1970utc'], reverse=True)
    else:
        file_list.sort(key=lambda x: x['mtime_sec_since_1970utc'], reverse=False)
   
    return file_list


def file_id_list(dir, ext, n_tail_to_cut_off):
    
    if not os.path.exists(dir):
        return []
    
    image_files = [f for f in os.listdir(dir) if f.endswith(ext)]

    # remove file_ending and _0000
    image_files = [f[0:-n_tail_to_cut_off] for f in image_files]
    
    return sorted(list(set(image_files)))


# Example usage
if __name__ == "__main__":
    from config import config
    directory_path = '/gpfs/projects/KimGroup/data/mic-mkfz/results/Dataset105_CBCTBladderRectumBowel/nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0/'
    files = list_files(directory_path, include_sub_folders=False)  # Only .py files
    print(f'num of files = {len(files)}')
    import json
    for file in files:  # Combining with your previous question—first 6 files
        print(json.dumps(file, indent=4))