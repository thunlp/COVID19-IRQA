def get_class(name):
    if name in ["bert2tag", "roberta2tag"]:
        return build_bert2tag_dataset, batchify_Bert2tag_features_for_train, batchify_Bert2tag_features_for_test
    elif name in ["bert2gram", "roberta2gram"]:
        return build_bert2gram_dataset, batchify_Bert2Gram_features_for_train, batchify_Bert2Gram_features_for_test
    raise RuntimeError('Invalid retriever class: %s' % name)
    

from .loader_utils import load_dataset
from .bert2tag_dataloader import build_bert2tag_dataset, batchify_Bert2tag_features_for_train, batchify_Bert2tag_features_for_test
from .bert2gram_dataloader import build_bert2gram_dataset, batchify_Bert2Gram_features_for_train, batchify_Bert2Gram_features_for_test