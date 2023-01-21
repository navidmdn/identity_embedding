import pickle
from typing import List, Dict, Union

from tqdm import tqdm
import torch


def load_pickled_data(path) -> Union[List[str], List[List[str]]]:
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def build_vocab(list_token_list: List[List[str]]) -> Dict:
    result_dict = {}
    for li in list_token_list:
        for token in li:
            if token not in result_dict:
                result_dict[token] = len(result_dict)
    return result_dict



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



