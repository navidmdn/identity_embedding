import os

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from src.util import load_pickled_data, remove_file_or_dir
import argparse


def load_w2v_model(model_path):
    model = Word2Vec.load(model_path)
    return model


class Callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1


def train_w2v_model(sentences, model_path, epochs=300, vector_size=768, window=8, min_count=1, workers=4, sg=1,
                    negative=8, seed=1, compute_loss=True):
    os.makedirs('./models', exist_ok=True)
    monitor_loss_cb = [Callback(), ]
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=sg,
                     negative=negative, seed=seed, compute_loss=compute_loss, callbacks=monitor_loss_cb, epochs=epochs)

    remove_file_or_dir(model_path)
    model.save(model_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--vector_size', type=int, default=768)
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--min_count', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--sg', type=int, default=1)
    parser.add_argument('--negative', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--compute_loss', type=bool, default=True)
    args = parser.parse_args()

    sentences = load_pickled_data(args.input)
    model = train_w2v_model(sentences, args.output, epochs=args.epochs, vector_size=args.vector_size, window=args.window,
                            min_count=args.min_count, workers=args.workers, sg=args.sg, negative=args.negative,
                            seed=args.seed, compute_loss=args.compute_loss)
    return model


if __name__ == '__main__':
    main()
