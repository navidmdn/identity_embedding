import pickle
from collections import OrderedDict


def load_bios(dataset='twitter', mode='tests'):
    with open(f'data/{dataset}_{mode}_bios.pkl', 'rb') as f:
        bios = pickle.load(f)

    # filter only bios with more than 1 PI
    atleast_2pi = [x for x in bios if len(x) > 1]
    return atleast_2pi


def build_restricted_target_dataset(bios, vocab, generalization):
    test_ds = []

    for bio in bios:
        for idx, pi in enumerate(bio):
            if pi in vocab:
                remaining = [x for x in bio if x != pi]

                # we want the remainig not to be available in vocab to tests generalization
                if generalization:
                    gen = True
                    for rem in remaining:
                        if rem in vocab:
                            gen = False

                    if not gen:
                        continue
                else:
                    unk = True
                    for rem in remaining:
                        if rem in vocab:
                            unk = False
                    if unk:
                        continue

                remaining_ctxt = ', '.join(remaining)
                if len(remaining) == 0:
                    continue
                test_ds.append((remaining, remaining_ctxt, pi))

    return test_ds


def create_restricted_target_test_dataset(dataset, vocab, generalization=False):
    print(f"creating dataset for {dataset}:")

    test_bios = load_bios(dataset)
    print(f"total tests bios: {len(test_bios)}")

    filtered_test_bios = []
    for bio in test_bios:
        for pi in bio:
            if pi in vocab:
                filtered_test_bios.append(bio)
                break

    print(f"total tests bios after restriction: {len(filtered_test_bios)}")

    test_ds = build_restricted_target_dataset(filtered_test_bios, vocab, generalization)
    print(f"total tests dataset entires: ", len(test_ds))
    print(f"vocab size:", len(vocab))
    return test_ds, vocab



