import json
import os

names = [
    'G_movie', 'G_news', 'G_twitter', 'P_Andersen', 'P_Conan', 'P_Dickens', 'P_Shakespeare', 'P_Tolstoy',
    'T_administration', 'T_animals', 'T_anime', 'T_architecture', 'T_art', 'T_astronomy', 'T_automotive',
    'T_beauty', 'T_biology', 'T_chemistry', 'T_civil', 'T_computer_science', 'T_consumption', 'T_culture',
    'T_current_affairs', 'T_dance', 'T_design', 'T_drinks', 'T_earth_sciences', 'T_economy', 'T_education',
    'T_emotion', 'T_engineering', 'T_entertainment', 'T_family', 'T_finance', 'T_food', 'T_game',
    'T_geography', 'T_gourmet', 'T_health', 'T_history', 'T_humanities', 'T_interpersonal', 'T_judicial',
    'T_language', 'T_law', 'T_life', 'T_literature', 'T_love', 'T_mathematics', 'T_medical', 'T_military',
    'T_movies', 'T_music', 'T_nature', 'T_pets', 'T_philosophy', 'T_physics', 'T_plants', 'T_politics',
    'T_psychology', 'T_real_estate', 'T_regulations', 'T_religion', 'T_shopping', 'T_society', 'T_sport',
    'T_technology', 'T_TV', 'T_war', 'T_work'
]

for name in names:
    json_file_path = './data_stego/LSCS/indiv/' + name + '/' + name + '.json'
    output_file_path = './data_stego/LSCS/indiv/' + name + '/1.txt'

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for text in data['covertexts']:
            text_with_literal_newlines = text.replace('\n', '\\n').replace('\n\n', '\\n\\n')
            output_file.write(text_with_literal_newlines + '\n')

    txt_file_path = output_file_path
    modified_file_path = './data_stego/LSCS/indiv/' + name + '/' + name + '.txt'

    with open(txt_file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    modified_text = text.replace('\\n\\n', ' ').replace('\\n', '')
    with open(modified_file_path, 'w', encoding='utf-8') as modified_file:
        modified_file.write(modified_text)

    os.remove(output_file_path)



# import os
# import json
#
# all_covertexts = []
# root_dir = './data_stego/LSCS/'
# for dir_name in os.listdir(root_dir):
#     dir_path = os.path.join(root_dir, dir_name)
#     if os.path.isdir(dir_path):
#         for file_name in os.listdir(dir_path):
#             if file_name.endswith('.json'):
#                 file_path = os.path.join(dir_path, file_name)
#                 with open(file_path, 'r') as f:
#                     data = json.load(f)
#                     if 'covertexts' in data:
#                         all_covertexts.extend(data['covertexts'])
#
# merged_json = {'covertexts': all_covertexts}
#
# with open('./data_stego/LSCS/LSCS.json', 'w') as f:
#     json.dump(merged_json, f, indent=4)
#



# import random
# import os
# import shutil
#
#
# def split_dataset(input_file, train_ratio=0.8):
#     with open(input_file, 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#     random.shuffle(lines)
#     total_samples = len(lines)
#     train_samples = int(total_samples * train_ratio)
#     train_data = lines[:train_samples]
#     test_data = lines[train_samples:]
#
#
#     train_file = input_file.replace('.txt', '_train.txt')
#     test_file = input_file.replace('.txt', '_test.txt')
#
#     with open(train_file, 'w', encoding='utf-8') as train_file:
#         train_file.writelines(train_data)
#
#     with open(test_file, 'w', encoding='utf-8') as test_file:
#         test_file.writelines(test_data)
#
#     print(f'Dataset split successful! Train samples: {len(train_data)}, Test samples: {len(test_data)}')
#
#
# input_file_path = './data_stego/LSCS/A_Overall/cover.txt'
# split_dataset(input_file_path)
