from sentence_transformers import SentenceTransformer
from numpy import linalg as LA
import numpy as np


new_projection_measures = [

    {
        "group":
            "age",
        "names": ['young', 'old'],
        "sets": [['15 years old', '18 years old', 'sophomore in college', 'student at', "umich'22", '18', '21'],
                 ['retired person', 'I’m old', '50 years old', '65 years old', '61yr old', 'grandparent of',
                  'old man', 'old woman', 'grandma to', 'grandpa to', 'tenured', 'long career']],
        "paper":
            "this_long",
        "is_paired":
            True
    },
    {
        "group":
            "partisanship",
        "names": ['democrat', 'republican'],
        "sets": [['pro socialism', 'liberal democrat', 'never trump', 'proud democrat', 'vote blue no matter who',
                  '#resist', '#voteblue', '#nevertrump', 'left leaning', '#democraticdownballot',
                    '#notvotebluenomatterwho', '#bidenharris', '#resist', '#bluewave', '#democraticsocialist'],
                 ['right leaning', 'trump won', 'never biden', 'fuck biden', '#maga', '#kag', 'Trump conservative',
                  'conservative and America First', 'proud Trump supporter', 'trump fan', '#MAGA Republican',
                  'constitutional conservative patriot', '#trump2024']],
        "paper":
            "unk",
        "is_paired":
            True
    },

    {
        "group":
            "religion",
        "names": ['atheist', 'religious'],
        "sets": [['atheist', 'nonbeliever', 'proud atheist', 'totally secular', '#cancelreligion'],
                 ['Catholic', 'jesus christ', 'follower of christ', 'priest', 'lover of jesus', 'christian episcopalian',
                  'jesus loving christian', 'john 3:16', 'gospel of the lord jesus christ', 'minister at united church',
                  'christ-follower', 'god first', 'isaiah 55:6', 'woman of faith', 'man of faith']],
        "paper":
            "unk",
        "is_paired":
            True
    },
    {
        "group":
            "gender",
        "names": ['woman', 'man'],
        "sets": [[
            'sister', 'wife', 'mother', 'Proud Mama and Wife', 'grandmother of', 'mother of one', 'mama of one',
            'wife of', 'Loving Wife', 'she', 'her', 'hers'
        ],
            [
                'husband to', 'brother', 'husband', 'father', 'grandfather of one', 'father of one', 'Loving husband',
                'he', 'him', 'his', 'son', 'brother', 'brother-in-law', 'uncle', 'nephew'
            ]],
        "paper":
            "bolukbasi_words",
        "is_paired":
            True
    },
    {
        "group":
            "politics",
        "names": ['apolotical', 'political'],
        "sets": [
            [
                'football fan', 'soccer fan', 'hockey fan', 'all about sports', 'all about music', 'art and culture',
                'looking for love', 'marketing', 'bitcoin cryptocurrency', 'music lover', 'alternative rock',
                'science writer', 'poetry', 'anime, cartoons', 'nascar', 'full-stack javascript software engineer',
                'musician, artist', 'football coach', 'movie lover', 'music producer', 'digital marketer, marketing strategist',
                'market analyst, trader, equities', 'celebrity and entertainment news', 'telling stories, making films',
                'blockchain', 'cooking, books', 'theatre, opera, music, art', 'software developer', 'bitcoin',
                'seo specialist', 'film maker'
            ],
            [
                'pro socialism', 'liberal democrat', 'never trump', 'proud democrat', 'vote blue no matter who',
                  '#resist', '#voteblue', '#nevertrump', 'left leaning', '#democraticdownballot',
                    '#notvotebluenomatterwho', '#bidenharris', '#resist', '#bluewave', '#democraticsocialist',
                 'right leaning', 'trump won', 'never biden', 'fuck biden', '#maga', '#kag', 'Trump conservative',
                  'conservative and America First', 'proud Trump supporter', 'trump fan', '#MAGA Republican',
                  'constitutional conservative patriot', '#trump2024'

            ]],
        "paper":
            "bolukbasi_words",
        "is_paired":
            True
    }
]

old_projection_measures = [

    {
        "group":
            "age",
        "names": ['young', 'old'],
        "sets": [['young', 'new', 'youthful', 'young'],
                 ['old', 'old', 'elderly', 'aged']],
        "paper":
            "this_long",
        "is_paired":
            True
    },
    {
        "group":
        "partisanship",
        "names": ['democrat', 'republican'],
        "sets": [['democratic party supporter', 'left-leaning', 'democrat'], ['republican party supporter', 'right-leaning', 'republican']],
        "paper":
        "unk",
        "is_paired":
        True
    },
    {
        "group":
        "religion",
        "names": ['atheist', 'religious'],
        "sets": [['atheistic', 'agnostic', 'non-believing', 'skeptical'], ['religious', 'faithful', 'christian', 'believe in lord']],
        "paper":
        "unk",
        "is_paired":
        True
    },
    {
        "group":
        "gender",
        "names": ['woman', 'man'],
        "sets": [
            ['mother of x', 'grand mother'], ['father of x', 'grand father']
        ],
        "paper":
        "bolukbasi_words",
        "is_paired":
        True
    },
    {
        "group":
            "politics",
        "names": ['apolotical', 'political'],
        "sets": [[
            'music', 'sports', 'culture', 'tech'
        ],
            [
                'politics', 'political', 'democrat', 'republican'
            ]],
        "paper":
            "bolukbasi_words",
        "is_paired":
            True
    }
]


def load_sbert_based_model(model_name, device='cpu'):
    model = SentenceTransformer(model_name, device=device)
    model.eval()
    return model


def load_dimensions_dict(measures):
    dims = {}
    for m in measures:
        if m['group'] in dims:
            dims[m['group']][0].extend(m['sets'][0])
            dims[m['group']][1].extend(m['sets'][1])
        else:
            dims[m['group']] = [m['sets'][0], m['sets'][1]]

    for g, p in dims.items():
        p[0] = list(set(p[0]))
        p[1] = list(set(p[1]))

    return dims


def normalize(wv):
    # normalize vectors
    norms = np.apply_along_axis(LA.norm, 1, wv)
    wv = wv / (norms[:, np.newaxis] + 1e-6)
    return wv


def ripa(w, b):
    return w.dot(b) / LA.norm(b)


def get_proj_embeddings(init_embs, pole_diffs):
    pi_prj_embs = []
    print("calculating projections")
    for init_emb in init_embs:
        prj = []
        for diff in pole_diffs:
            prj.append(ripa(init_emb, diff))
        pi_prj_embs.append(np.array(prj))

    return pi_prj_embs


def project_embeddings(model, dims, sentence_embeddings):
    # get pole diffs
    pole_diffs = []

    for pol_name, poles in dims.items():
        vs = model.encode([', '.join(poles[0]), ', '.join(poles[1])])
        pole_diffs.append(vs[1] - vs[0])

    # project to poles
    projected = []

    for init_emb in sentence_embeddings:
        prj = []
        for diff in pole_diffs:
            prj.append(ripa(init_emb, diff))
        projected.append(np.array(prj))

    return projected


def get_sentence_projections(sentences, model, batch_size=128, device='cpu', show_progress_bar=True, as_dict=False,
                             measures=None, projection_measures=old_projection_measures):
    print("Getting sentence embeddings")
    sentence_embeddings = model.encode(
        sentences, batch_size=batch_size, show_progress_bar=show_progress_bar, device=device)

    dim_dict = load_dimensions_dict(projection_measures)
    print(dim_dict)
    if measures is not None:
        for k, v in measures.items():
            dim_dict[k] = v

    print("Projecting embeddings")
    projections = project_embeddings(model, dim_dict, sentence_embeddings)

    if as_dict:
        results = []
        for prj in projections:
            results.append({k: v for k, v in zip([m['group'] for m in projection_measures], prj)})

        return results

    return projections



