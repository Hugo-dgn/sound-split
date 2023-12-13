#!/bin/bash

mkdir dataset
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xvf dev-clean.tar.gz -C dataset

# Remove tar.gz files
rm dev-clean.tar.gz

echo "LibriSpeech dataset downloaded and extracted to dataset folder."
