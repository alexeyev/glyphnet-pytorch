# glyphnet-pytorch
...a.k.a. cracking **MNIST: Ancient Egypt Edition**

This repository presents a custom (non-official) PyTorch-based implementation of the **Glyphnet** 
classifier introduced in the work [A Deep Learning Approach 
to Ancient Egyptian Hieroglyphs 
Classification, 2021](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9528382) 
and apply it to the data accompanying the work ["Automatic Egyptian 
Hieroglyph Recognition by Retrieving Images as Texts", 2013](https://jvgemert.github.io/pub/FrankenACMMM13egyptian.pdf) 
(NB! GlyphNet paper uses a larger dataset).

We hope that this implementation of the model will encourage
the further research in this direction.

## Requirements

Please see `requirements.txt`.

## Setting everything up

An entry point is the script `prepare_data.sh` that downloads the dataset and splits it into train/test 
parts in a 'stratified' manner, i.e. keeping all labels with just a single image in the training set, 
yet preserving similar label counts distributions in each part of the dataset. 

It should print

    DEBUG:root:Labels total: 172
    DEBUG:root:Labels seen just once: 37

before shutting down.

## Notes

* Please do not confuse this work with another [GlyphNet project](https://github.com/noahtren/GlyphNet) 
training networks to communicate using a visual language
