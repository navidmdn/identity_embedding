from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from src.rank import create_restricted_target_test_dataset

import numpy as np


def load_w2v_model(path):
    return Word2Vec.load(path)


def load_sbert_based_model(path, device='cpu'):
    model = SentenceTransformer(path, device=device)
    model.eval()
    return model


def load_bert_based_model(path, device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    model = AutoModel.from_pretrained(path)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_model(config):
    tokenizer = None
    if config.base_model == 'w2v':
        model = load_w2v_model(config.model_path)
    elif config.base_model == 'sbert':
        model = load_sbert_based_model(config.model_path, config.device)
    elif config.base_model == 'bert':
        model, tokenizer = load_bert_based_model(config.model_path, config.device)
    else:
        raise NotImplementedError()

    return model, tokenizer


def get_vocab_from_w2v_model(path):
    model = load_w2v_model(path)
    return model.wv.key_to_index


def load_dataset(config):
    test_ds, vocab = create_restricted_target_test_dataset(
        dataset=config.dataset,
        vocab=get_vocab_from_w2v_model(config.w2v_model_path),
        generalization=config.generalization,
    )

    if config.sample_size > 0:
        sample_indices = np.random.randint(0, len(test_ds), config.sample_size)
        test_ds = [test_ds[i] for i in sample_indices]

    if config.base_model == 'w2v':
        test_ds = [[x[0], x[2]] for x in test_ds]
    else:
        test_ds = [[x[1], x[2]] for x in test_ds]

    return test_ds, vocab






