import pickle
from typing import List, Dict, Union
import os
import shutil


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


def remove_file_or_dir(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path, ignore_errors=True)
            os.remove(path)
        except OSError as e:
            pass




