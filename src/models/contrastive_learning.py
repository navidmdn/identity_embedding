from typing import List, Dict, Tuple
from collections import Counter
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers import evaluation
from torch.utils.data import DataLoader
from src.util import load_pickled_data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from random import randint


class Triple:
    def __init__(self, anchor, positive, negative):
        self.anchor = anchor
        self.positive = positive
        self.negative = negative

    def __repr__(self):
        return f"Triple(anchor={self.anchor}, positive={self.positive}, negative={self.negative})"

    def __eq__(self, other):
        return self.anchor == other.anchor and self.positive == other.positive and self.negative == other.negative


def build_neighborhood_dict(list_token_list: List[List[str]]) -> Dict:
    neighbors = {}
    for li in list_token_list:
        for token in li:
            if token not in neighbors:
                neighbors[token] = Counter()
            rest = [t for t in li if t != token]
            neighbors[token].update(rest)

    return neighbors


def generate_triples(list_token_list: List[List[str]], neighborhood_dict: Dict) -> List[Triple]:
    pi_set = list(neighborhood_dict.keys())

    samples = []
    for idx, bio in tqdm(enumerate(list_token_list), total=len(list_token_list)):
        if len(bio) != len(set(bio)):
            continue
        anchor = np.random.choice(bio, size=1, replace=False)[0]
        pos = [pi for pi in bio if pi != anchor]
        pos = ", ".join(pos)

        neg_idx = randint(0, len(pi_set) - 1)
        while pi_set[neg_idx] in neighborhood_dict[anchor] or pi_set[neg_idx] == anchor:
            neg_idx = randint(0, len(pi_set) - 1)

        samples.append(Triple(anchor, pos, pi_set[neg_idx]))
    return samples


def prepare_train_set(triples: List[Triple]) -> List[InputExample]:
    train_examples = []
    for triple in triples:
       train_examples.append(InputExample(texts=[triple.anchor, triple.positive], label=1.0))
       train_examples.append(InputExample(texts=[triple.anchor, triple.negative], label=0.0))
    return train_examples


#TODO: set return type in function signature
def get_evaluator(evaluation_triples: List[Triple]):
    sent1s = []
    sent2s = []
    scores = []

    for triple in evaluation_triples:
        sent1s.append(triple.anchor)
        sent1s.append(triple.anchor)

        sent2s.append(triple.positive)
        sent2s.append(triple.negative)

        scores.append(1.0)
        scores.append(0.0)

    return evaluation.EmbeddingSimilarityEvaluator(sent1s, sent2s, scores)


def fine_tune(base_model_name, train_bios, batch_size=256, epochs=1, evaluation_steps=5000, warmup_steps=5000,
              checkpoint_save_steps=35000, output_path='contrastive_learning_model'):

    pi_neighborhoods = build_neighborhood_dict(train_bios)

    triples = generate_triples(train_bios, pi_neighborhoods)

    train_triples, validation_triples = train_test_split(triples, test_size=0.1)

    print(f"Number of train samples: {len(train_triples)}")
    print(f"Number of validation samples: {len(validation_triples)}")

    train_examples = prepare_train_set(train_triples)

    model = SentenceTransformer(base_model_name)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=get_evaluator(validation_triples),
        epochs=epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        checkpoint_save_steps=checkpoint_save_steps,
        output_path=output_path,
    )

    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--evaluation_steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--checkpoint_save_steps", type=int, default=35000)
    parser.add_argument("--output_path", type=str, default='contrastive_learning_model')
    args = parser.parse_args()

    train_bios = load_pickled_data(args.train_data_path)

    fine_tune(args.base_model_name, train_bios, args.batch_size, args.epochs, args.evaluation_steps,
              args.warmup_steps, args.checkpoint_save_steps, args.output_path)


if __name__ == '__main__':
    main()
