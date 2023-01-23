from datasets import Dataset
import pandas as pd
import collections
import numpy as np
from transformers import TrainingArguments
from transformers import Trainer
from transformers.data.data_collator import default_data_collator
from typing import List
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from src.util import load_pickled_data

import shutil
import os


def load_clm_model_and_tokenizer(model_name: str, tokenizer_name: str = None):
    if tokenizer_name is None:
        tokenizer_name = model_name

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


def whole_pi_masking_data_collator(tokenizer):
    def _whole_pi_masking_data_collator(features, tokenizer=tokenizer):
        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)

            for idx, (token_id, label) in enumerate(zip(input_ids, labels)):
                if token_id == tokenizer.mask_token_id:
                    new_labels[idx] = label
            feature["labels"] = new_labels
        return default_data_collator(features)

    return _whole_pi_masking_data_collator


def prepare_train_dataset(bios: List[List[str]], tokenizer, max_length: int = 80,
                          max_train_samples: int = int(1e6), max_val_samples: int = int(1e4)):
    train_df = pd.DataFrame([{'bios': ", ".join(t), "bios_l": t} for t in bios])
    train_dataset = Dataset.from_pandas(train_df)

    # before masking we need to tokenize the bios and get the label ids
    label_ids = tokenizer(train_dataset["bios"], padding='max_length', max_length=max_length, truncation=True)["input_ids"]

    def preprocess(examples):
        # randomly mask one pi in each bio
        masked_examples = []
        for bio in examples["bios_l"]:
            masked_bio = bio.copy()
            masked_idx = np.random.choice(len(masked_bio), size=1, replace=False)[0]
            pi = masked_bio[masked_idx]
            pi_ids = tokenizer([pi, ])["input_ids"][0]
            masked_pi = "[MASK]" * (len(pi_ids)-2)
            masked_bio[masked_idx] = masked_pi
            masked_examples.append(", ".join(masked_bio))

        examples['bios'] = masked_examples

        result = tokenizer(examples["bios"], padding='max_length', max_length=80, truncation=True)
        result["labels"] = label_ids
        return result

    tokenized_dataset = train_dataset.map(
        preprocess, batched=True, remove_columns=["bios", "bios_l"], batch_size=250, num_proc=1
    )

    sampled_dataset = tokenized_dataset.train_test_split(
        train_size=max_train_samples, test_size=max_val_samples, shuffle=True)

    return sampled_dataset


#it should call prepare_train_dataset and then train the model
def fine_tune_masked_lm(bios: List[List[str]], model_name: str, tokenizer_name: str = None, max_length: int = 80,
                        max_train_samples: int = int(1e6), max_val_samples: int = int(1e4), epochs: int = 1,
                        batch_size: int = 8, output_dir: str = "./results", ):

    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            os.remove(output_dir)
        except OSError as e:
            pass

    model, tokenizer = load_clm_model_and_tokenizer(model_name, tokenizer_name)
    train_val_dataset = prepare_train_dataset(bios, tokenizer, max_length, max_train_samples, max_val_samples)

    logging_steps = max(1, int(len(train_val_dataset["train"]) / batch_size / 10))
    args = TrainingArguments(
        # output_dir: directory where the model checkpoints will be saved.
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=logging_steps,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=logging_steps,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="tensorboard",
        weight_decay=0.01,
        remove_unused_columns=False

    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_val_dataset["train"],
        eval_dataset=train_val_dataset["test"],
        data_collator=whole_pi_masking_data_collator(tokenizer),
    )

    trainer.train()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=80)
    parser.add_argument("--max_train_samples", type=int, default=int(1e6))
    parser.add_argument("--max_val_samples", type=int, default=int(1e4))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    # load bios
    bios = load_pickled_data(args.input_file)

    fine_tune_masked_lm(bios, args.model_name, args.tokenizer_name, args.max_length, args.max_train_samples,
                        args.max_val_samples, args.epochs, args.batch_size, args.output_dir)


if __name__ == "__main__":
    main()
