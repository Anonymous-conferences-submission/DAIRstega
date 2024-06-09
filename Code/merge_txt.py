import os

def merge_all_txt_files(output_file, folder_path):
    with open(output_file, 'w', encoding='utf-8') as output:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as input_file:
                        output.write(input_file.read())

merge_all_txt_files('./data_cover/A_Overall.txt', './data_cover')

