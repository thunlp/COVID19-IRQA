from tqdm import tqdm
import json
import os
import codecs


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    f.close()
    return examples


def load_txt_to_list(file_path, spliter='\t', terminator="\n"):
    """ Load file.txt by lines, convert it to list. """
    txt_list = []
    with open(file=file_path, mode='r', encoding='utf-8') as fi:
        lines = fi.readlines()
        for i, line in enumerate(tqdm(lines)):
            line = line.strip(terminator).split(spliter)
            txt_list.append(line)
    return txt_list


def load_and_merge_json(folder_path, filenames):
    """ load and merge json files in folder_path. """
    all_files = []
    for filename in filenames:
        filename = os.path.join(folder_path, filename)
        file = json.load(open(filename, 'r', encoding='utf-8'))
        all_files.append(file)
    return all_files



def load_jsonl(file_path):
    """ Load file.jsonl ."""
    data_list = []
    with codecs.open(file_path, mode='r', encoding='utf-8') as fi:
        for idx, line in enumerate(tqdm(fi)):
            jsonl = json.loads(line)
            data_list.append(jsonl)
    return data_list