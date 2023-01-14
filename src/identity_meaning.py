from gensim.models import Word2Vec
import pickle
from collections import OrderedDict


def load_w2v_model(model_path):
    model = Word2Vec.load(model_path)
    return model


def load_bios(dataset='twitter', mode='test'):
    with open(f'data/{dataset}_{mode}_bios.pkl', 'rb') as f:
        bios = pickle.load(f)

    # filter only bios with more than 1 PI
    atleast_2pi = [x for x in bios if len(x) > 1]
    return atleast_2pi

def build_restricted_target_dataset(bios, vocab, in_domain):
    test_ds = []

    for bio in bios:
        for idx, pi in enumerate(bio):
            if pi in vocab:
                remaining = [x for x in bio if x != pi]

                # we want the remainig not to be available in vocab to test generalization
                if not in_domain:
                    gen = True
                    for rem in remaining:
                        if rem in vocab:
                            gen = False

                    if not gen:
                        continue

                remaining_ctxt = ', '.join(remaining)
                if len(remaining) == 0:
                    continue
                test_ds.append((remaining, remaining_ctxt, pi))

    return test_ds


def create_restricted_target_test_dataset(dataset, vocab, in_domain=True):
    print(f"creating dataset for {dataset}:")

    test_bios = load_bios(dataset, mode='test')
    print(f"total test bios: {len(test_bios)}")

    # each test instance should contain at least one PI which is seen during training time
    filtered_test_bios = []
    for bio in test_bios:
        for pi in bio:
            if pi in vocab:
                filtered_test_bios.append(bio)
                break

    print(f"total test bios after restriction: {len(filtered_test_bios)}")

    test_ds = build_restricted_target_dataset(filtered_test_bios, vocab, in_domain)
    print(f"total test dataset entires: ", len(test_ds))

    target_pis = set()
    for _, _, y in test_ds:
        target_pis.add(y)

    pi_dict = OrderedDict()
    for p in target_pis:
        pi_dict[p] = len(pi_dict)

    print(f"vocab size: {len(pi_dict)}")
    return test_ds, pi_dict


