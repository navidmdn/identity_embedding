import pickle
from typing import List, Dict


def load_pickled_data(path) -> List[str]:
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


