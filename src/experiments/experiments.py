from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from src.rank import create_restricted_target_test_dataset, calculate_rankings

import argparse
import numpy as np
import yaml
import os


root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    if config.fine_tuned:
        path = os.path.join(root_dir, config.model_path)
    else:
        path = config.model_path

    tokenizer = None
    if config.base_model == 'w2v':
        model = load_w2v_model(path)
    elif config.base_model == 'sbert':
        model = load_sbert_based_model(path, config.device)
    elif config.base_model == 'bert':
        model, tokenizer = load_bert_based_model(path, config.device)
    else:
        raise NotImplementedError()

    return model, tokenizer


def get_vocab_from_w2v_model(path):
    model = load_w2v_model(path)
    return model.wv.key_to_index


def load_dataset(config):
    w2v_path = os.path.join(root_dir, config.w2v_model_path)
    test_ds, vocab = create_restricted_target_test_dataset(
        base_path=os.path.join(root_dir, 'data'),
        dataset=config.dataset,
        vocab=get_vocab_from_w2v_model(w2v_path),
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


def load_config(path):
    class Config:
        def __init__(self, dictionary):
            self.dictionary = dictionary

        def __getattr__(self, item):
            if item not in self.dictionary:
                return None
            return self.dictionary[item]

        def __repr__(self):
            return str(self.dictionary)

    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return Config(config)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config file')
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"config: {config}")

    model, tokenizer = load_model(config)
    test_ds, vocab = load_dataset(config)

    print(f"test dataset size: {len(test_ds)}")

    target_ranks, softmax_scores = calculate_rankings(
        model=model,
        dataset=test_ds,
        pi_dict=vocab,
        device=config.device,
        cosine_bs=config.cosine_bs,
        emb_bs=config.emb_bs,
        default_encoder=config.base_model == 'sbert',
        w2v=config.base_model == 'w2v',
        tokenizer=tokenizer,
        average_k_layers=config.average_k_layers,
    )

    print(f"average rank: {np.mean(target_ranks)}")
    print(f"average softmax score: {np.mean(softmax_scores)}")
    print(f"top100-accuracy: {np.mean(np.array(target_ranks) <= 100)}")


if __name__ == '__main__':
    main()

