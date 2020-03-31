PAD = 0
UNK = 100
BOS = 101
EOS = 102
DIGIT = 1

PAD_WORD = '[PAD]'
UNK_WORD = '[UNK]'
BOS_WORD = '[CLS]'
EOS_WORD = '[SEP]'
DIGIT_WORD = 'DIGIT'

Tag2Idx = {'O':0, 'B':1, 'I':2, 'E':3, 'U':4}
Idx2Tag = ['O', 'B', 'I', 'E', 'U']


Decode_Candidate_Number = {'comm_use_subset':10, 'noncomm_use_subset':10, 'custom_license':10, 'biorxiv_medrxiv':10}

class IdxTag_Converter(object):
    ''' idx2tag : a tag list like ['O','B','I','E','U']
        tag2idx : {'O': 0, 'B': 1, ..., 'U':4}
    '''
    def __init__(self, idx2tag):
        self.idx2tag = idx2tag
        tag2idx = {}
        for idx, tag in enumerate(idx2tag):
            tag2idx[tag] = idx
        self.tag2idx = tag2idx
        
    def convert_idx2tag(self, index_list):
        tag_list = [self.idx2tag[index] for index in index_list]
        return tag_list
        
    def convert_tag2idx(self, tag_list):
        index_list = [self.tag2idx[tag] for tag in tag_list]
        return index_list

# 'O' : non-keyphrase
# 'B' : begin word of the keyphrase
# 'I' : middle word of the keyphrase
# 'E' : end word of the keyphrase
# 'U' : single word keyphrase