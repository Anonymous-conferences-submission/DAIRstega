import os
import shutil

def copy_txt_to_all_folders(txt_file, base_folder):
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

    for folder in subfolders:
        destination_path = os.path.join(folder, os.path.basename(txt_file))
        shutil.copy(txt_file, destination_path)

copy_txt_to_all_folders('1.txt', './data_stego/7b-48')
