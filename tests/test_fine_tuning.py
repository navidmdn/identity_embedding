import pytest
from src.w2v import train_w2v_model
from src.masked_pi_modeling import prepare_train_dataset, fine_tune_masked_lm, load_clm_model_and_tokenizer,\
    whole_pi_masking_data_collator
from src.contrastive_learning import generate_triples, build_neighborhood_dict, Triple, fine_tune
from datasets import DatasetDict
import numpy as np
import torch
import random

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

params = [
    {"epochs": 1, "model_path": "/tmp/results"},
]
@pytest.mark.parametrize("bios", [[['singer', 'worker', 'professor'], ['teacher', 'father'],]])
@pytest.mark.parametrize("train_params", params)
def test_w2v_training(bios, train_params):
    epoch = train_params['epochs']
    model_path = train_params['model_path']
    res = train_w2v_model(bios, epochs=epoch, model_path=model_path,)
    assert res is not None


params = [
    {"epochs": 1, "model_path": "/tmp/results", "model_name": "bert-base-uncased"}
]
@pytest.mark.parametrize("bios", [[['vocal singer', 'worker', 'professor'], ['teacher', 'father'],]])
@pytest.mark.parametrize("train_params", params)
def test_mlm_data_preparation(bios, train_params):
    model_name = tokenizer_name = train_params['model_name']
    _, tokenizer = load_clm_model_and_tokenizer(model_name, tokenizer_name)
    dataset = prepare_train_dataset(bios, tokenizer,
                                    max_length=80, max_train_samples=1, max_val_samples=1)

    assert isinstance(dataset, DatasetDict)
    samples = [dataset["train"][0], ]
    instances = whole_pi_masking_data_collator(tokenizer)(samples)
    input_ids = instances['input_ids'][0]
    labels = instances['labels'][0]
    assert tokenizer.decode(input_ids, skip_special_tokens=True) == ', worker, professor'
    assert input_ids[1] == input_ids[2] == tokenizer.mask_token_id
    assert labels[1] == tokenizer.convert_tokens_to_ids('vocal')
    assert labels[2] == tokenizer.convert_tokens_to_ids('singer')


params = [
    {"epochs": 1, "model_path": "/tmp/results", "model_name": "bert-base-uncased"}
]
@pytest.mark.parametrize("bios", [[['vocal singer', 'worker', 'professor'], ['teacher', 'father'],]])
@pytest.mark.parametrize("train_params", params)
def test_mlm_fine_tuning(bios, train_params):
    model_name = tokenizer_name = train_params['model_name']
    fine_tune_masked_lm(bios, model_name, tokenizer_name, epochs=train_params['epochs'], output_dir=train_params['model_path'],
                        batch_size=1, max_train_samples=1, max_val_samples=1)


@pytest.mark.parametrize("bios", [[['vocal singer', 'worker', 'professor'], ['worker', 'father'],
                                   ['professor', 'teacher'],]])
def test_contrastive_learning_triplet_generation(bios):
    np.random.seed(1)
    neighborhood_dict = build_neighborhood_dict(bios)
    triples = generate_triples(bios, neighborhood_dict)
    print(triples)
    assert triples[0] == Triple('vocal singer', 'worker, professor', 'teacher')
    assert triples[1] == Triple('father', 'worker', 'vocal singer')
    assert triples[2] == Triple('teacher', 'professor', 'vocal singer')


params = [
    {"epochs": 1, "batch_size": 1, "output_path": "/tmp/results", "base_model_name": "all-mpnet-base-v2"}
]
@pytest.mark.parametrize("bios", [[['vocal singer', 'worker', 'professor'], ['worker', 'father'],
                                   ['professor', 'teacher'],]])
@pytest.mark.parametrize("train_params", params)
def test_contrastive_learning_fine_tuning(bios, train_params):
    model = fine_tune(train_params['base_model_name'], bios, output_path=train_params['output_path'],
                      epochs=train_params['epochs'], batch_size=train_params['batch_size'])
    assert isinstance(model, torch.nn.Module)
