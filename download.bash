#!/bin/bash

mkdir -p train_dataset
mkdir -p test_dataset

if [ "$1" = "dev" ]; then
    TRAIN_DATASET_URL="https://www.openslr.org/resources/12/dev-clean.tar.gz"
    wget "$TRAIN_DATASET_URL"
    tar -xvf dev-clean.tar.gz -C train_dataset
else
    TRAIN_DATASET_URL="https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    wget "$TRAIN_DATASET_URL"
    tar -xvf train-clean-100.tar.gz -C train_dataset
fi

TEST_DATASET_URL="https://www.openslr.org/resources/12/test-clean.tar.gz"
wget "$TEST_DATASET_URL"
tar -xvf test-clean.tar.gz -C test_dataset

# Remove tar.gz files
rm *.tar.gz

echo "LibriSpeech dataset downloaded and extracted to dataset folder."
