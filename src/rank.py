from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, models, util
from torch.nn.functional import log_softmax
import pickle
import os


def get_bertbased_results_batched(model, tokenizer, str_l, bs=256, average_k_layers=1, device='cpu'):
    i = 0
    result = []
    pbar = tqdm(total=len(str_l))
    while i < len(str_l):
        batch = list(str_l[i:i + bs])
        with torch.no_grad():
            tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
            res_full = model(**tokens, output_hidden_states=True).hidden_states
            layers = []

            for k in range(-average_k_layers, 0):
                pooled_val = res_full[k].detach().cpu()
                # taking cls token embeddings
                layers.append(pooled_val[:, 0, :])

            stacked_layers = torch.stack(layers, dim=1)
            # print(stacked_layers.shape)

            # todo: based on reference paper this should be concatenation
            average_embs = torch.mean(stacked_layers, dim=1)
            # print(average_embs.shape)

            result.append(average_embs)
            i = i + bs
            pbar.update(bs)
    return torch.concat(result, dim=0)


def get_sbertbased_results_batched(model, str_l, bs=256):
    i = 0
    result = []
    pbar = tqdm(total=len(str_l))
    while i < len(str_l):
        batch = list(str_l[i:i + bs])
        with torch.no_grad():
            embeddings = model.encode(batch, convert_to_tensor=True)
            result.append(embeddings.detach().cpu())

            i = i + bs
            pbar.update(bs)
    return torch.concat(result, dim=0)


def normalize(word_vectors):
    norms = torch.norm(word_vectors, dim=1, keepdim=True)
    res = word_vectors / (norms + 1e-6)
    return res


def load_bios(base_path, dataset='twitter', mode='test'):
    with open(os.path.join(base_path, f'{dataset}_{mode}_bios.pkl'), 'rb') as f:
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

                # we want the remainig not to be available in vocab to test generalization
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


def create_restricted_target_test_dataset(base_path, dataset, vocab, generalization=False):
    print(f"creating dataset for {dataset}:")

    test_bios = load_bios(base_path, dataset)
    print(f"total test bios: {len(test_bios)}")

    filtered_test_bios = []
    for bio in test_bios:
        for pi in bio:
            if pi in vocab:
                filtered_test_bios.append(bio)
                break

    print(f"total test bios after restriction: {len(filtered_test_bios)}")

    test_ds = build_restricted_target_dataset(filtered_test_bios, vocab, generalization)
    print(f"total test dataset entires: ", len(test_ds))
    print(f"vocab size:", len(vocab))
    return test_ds, vocab


def calculate_rankings(model, dataset, pi_dict, device='cpu', cosine_bs=512, emb_bs=512,
                       default_encoder=False, w2v=False,
                       tokenizer=None, average_k_layers=1):

    pi_list = list(pi_dict.keys())
    X, ys = zip(*dataset)

    if w2v:
        emb_x = torch.tensor([model.wv.get_mean_vector(x) for x in X], device=device)
        emb_all = torch.tensor([model.wv.get_vector(x) for x in pi_list], device=device)
    elif default_encoder:
        emb_x = get_sbertbased_results_batched(model, X, bs=emb_bs)
        emb_all = get_sbertbased_results_batched(model, pi_list, bs=emb_bs)
    else:
        emb_x = get_bertbased_results_batched(model, tokenizer, X, average_k_layers=average_k_layers, bs=emb_bs)
        emb_all = get_bertbased_results_batched(model, tokenizer, pi_list, average_k_layers=average_k_layers, bs=emb_bs)

    emb_x = normalize(emb_x)
    emb_all = normalize(emb_all)

    print("calculating ranks...")

    i = 0
    pbar = tqdm(total=emb_x.shape[0])

    target_ranks = []
    softmax_scores = []

    while i < emb_x.shape[0]:
        batch = emb_x[i:i + cosine_bs]
        batch_y = ys[i:i + cosine_bs]
        batch_cosine_scores = util.cos_sim(batch.to(device), emb_all.to(device))
        batch_softmax = log_softmax(batch_cosine_scores, dim=1)
        ranks = torch.argsort(torch.argsort(batch_cosine_scores, dim=1, descending=True), dim=1)
        target_idxs = torch.tensor([pi_dict[y] for y in batch_y], dtype=torch.int64, device=device)

        batch_target_ranks = torch.gather(ranks, 1, target_idxs.unsqueeze(1).reshape(-1, 1)).type(torch.FloatTensor)
        batch_target_scores = torch.gather(batch_softmax, 1, target_idxs.unsqueeze(1).reshape(-1, 1)).type(
            torch.FloatTensor)

        target_ranks.append(batch_target_ranks)
        softmax_scores.append(batch_target_scores)

        i += cosine_bs
        pbar.update(cosine_bs)

    target_ranks = torch.concat(target_ranks)
    softmax_scores = torch.concat(softmax_scores)

    return target_ranks.reshape(-1).tolist(), softmax_scores.reshape(-1).tolist()


