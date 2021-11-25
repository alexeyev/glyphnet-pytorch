#!/usr/bin/env bash

wget http://iamai.nl/downloads/GlyphDataset.zip
unzip GlyphDataset.zip -d data
rm GlyphDataset.zip

rm -r prepared_data
mkdir prepared_data
python3 reader.py
#rm -r data