from typing import List, Dict, Tuple
from collections import Counter


def build_neighborhood_dict(list_token_list: List[List[str]]) -> Dict:
    neighbors = {}
    for li in list_token_list:
        for token in li:
            if token not in neighbors:
                neighbors[token] = Counter()
            rest = [t for t in li if t != token]
            neighbors[token].update(rest)

    return neighbors

def generate_triplets(list_token_list: List[List[str]]) -> List[Tuple[str, str, str]]:
    #TODO: implement after finding out the best method
    pass