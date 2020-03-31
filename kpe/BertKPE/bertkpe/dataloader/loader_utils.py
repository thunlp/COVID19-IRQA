import os
import json
import logging
import unicodedata
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

logger = logging.getLogger()

def load_dataset(preprocess_folder, dataset_class, name):
    logger.info("start loading %s %s data ..." %(dataset_class, name))
    filename = os.path.join(preprocess_folder, "%s.%s.json" % (dataset_class, name))
    
    with open(filename, "r", encoding="utf-8") as f:
        examples = json.load(f)
    f.close()
    logger.info("success loaded %s %s data : %d " 
                %(dataset_class, name, len(examples)))
    return examples


def flat_rank_pos(start_end_pos):
    flatten_postions = [pos for poses in start_end_pos for pos in poses]
    sorted_positions = sorted(flatten_postions, key=lambda x: x[0])
    return sorted_positions


def strict_filter_overlap(positions):
    '''delete overlap keyphrase positions for bert2tag'''
    previous_e = -1
    filter_positions = []
    for i, (s, e) in enumerate(positions):
        if s <= previous_e:
            continue
        filter_positions.append(positions[i])
        previous_e = e
    return filter_positions


def loose_filter_overlap(positions):
    '''delete overlap keyphrase positions for bert2attspan'''
    previous_s = -1
    filter_positions = []
    for i, (s, e) in enumerate(positions):
        if previous_s == s:
            continue
        elif previous_s < s:
            filter_positions.append(positions[i])
            previous_s = s
        else:
            logger.info('Error! previous start large than new start')
    return filter_positions



# Delete Over Scope keyphrase position (token_len > 510) and phrase_length > 5
def limit_scope_length(start_end_pos, valid_length, max_phrase_words):
    """filter out positions over scope & phase_length > 5"""
    filter_positions = []
    for positions in start_end_pos:
        _filter_position = [pos for pos in positions \
                            if pos[1] < valid_length and (pos[1]-pos[0]+1) <= max_phrase_words]
        if len(_filter_position) > 0:
            filter_positions.append(_filter_position)
    return filter_positions



def stemming(phrase):
    norm_chars = unicodedata.normalize('NFD', phrase)
    stem_chars = " ".join([stemmer.stem(w) for w in norm_chars.split(" ")])
    return norm_chars, stem_chars

