# identity_embedding

This repository contains code for the paper "X" (link to paper).

## Requirements

This project was tested on Ubuntu 20.04 with Python 3.8.5 and PyTorch 1.7.1. The required packages are listed in
`requirements.txt`.

## Getting data

The data used in the paper is available at [link to data]. Download the 
data and extract it to the `data` directory. There is the train and test dataset
for both Twitter and Wikipedia datasets.

## Training models from scratch

### Training the word2vec model for twitter

```bash
python train_word2vec.py \
 --input twitter_cleaned_train_bios_100.pkl 
 --output models/w2v_negsampling_twitter_100.model
```

### Training the bert model