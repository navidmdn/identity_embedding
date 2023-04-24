from sentence_transformers import SentenceTransformer
from numpy import linalg as LA
import numpy as np

projection_measures = [
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
        "politics",
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
        "education",
        "names": ['educated', 'uneducated'],
        "sets": [['educated', 'higher education'], ['uneducated', 'unschooled']],
        "paper":
        "unk",
        "is_paired":
        True
    },
    {
        "group":
        "employment status",
        "names": ['employed', 'unemployed'],
        "sets": [['employed', 'hired', 'working', 'on the job'], ['unemployed', 'jobless', 'out of work', 'retired']],
        "paper":
        "unk",
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


def get_sentence_projections(sentences, model, batch_size=128, device='cpu', show_progress_bar=True, as_dict=False):
    print("Getting sentence embeddings")
    sentence_embeddings = model.encode(
        sentences, batch_size=batch_size, show_progress_bar=show_progress_bar, device=device)

    print("Projecting embeddings")
    projections = project_embeddings(model, load_dimensions_dict(projection_measures), sentence_embeddings)

    if as_dict:
        results = []
        for prj in projections:
            results.append({k: v for k, v in zip([m['group'] for m in projection_measures], prj)})

        return results

    return projections



