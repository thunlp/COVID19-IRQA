from tqdm import tqdm
import json
import os

def save_jsonl(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as fo:
        for data in data_list:
            fo.write("{}\n".format(json.dumps(data)))
        fo.close()
    print("Success save file to %s \n" %filename)
    
    
    
def save_json(data_list, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_list, f)
    f.close()
    print("Success save file to %s \n" %filename)
    

def check_or_create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)