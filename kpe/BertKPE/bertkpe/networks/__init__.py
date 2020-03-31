def get_class(name):
    if name == "bert2tag":
        return BertForSeqTagging
    elif name == "roberta2tag":
        return RobertaForSeqTagging
    elif name == "bert2gram":
        return BertForCnnGramExtractor
    elif name == "roberta2gram":
        return RobertaForCnnGramExtractor
    raise RuntimeError('Invalid retriever class: %s' % name)

from .Bert2Tag import BertForSeqTagging
from .Roberta2Tag import RobertaForSeqTagging
from .Bert2Gram import BertForCnnGramExtractor
from .Roberta2Gram import RobertaForCnnGramExtractor
