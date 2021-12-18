# glyphnet-pytorch

This repository presents a custom (non-official) PyTorch-based implementation of the **Glyphnet** 
classifier introduced in the work [A Deep Learning Approach 
to Ancient Egyptian Hieroglyphs 
Classification, 2021](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9528382) 
and apply it to the data accompanying the work ["Automatic Egyptian 
Hieroglyph Recognition by Retrieving Images as Texts", 2013](https://jvgemert.github.io/pub/FrankenACMMM13egyptian.pdf) 
(NB! Glyphnet paper uses a larger dataset).


**Image** | ![aren't they pretty](/sample_images/230128_G17.png) | ![aren't they pretty](/sample_images/230067_G26.png) | ![aren't they pretty](/sample_images/230034_G25.png) 
------------ | ------------ | ------------- | -------------
**Gardiner code** | G17 | G26 | G25

We hope that this implementation of the model will encourage
the further research in this direction.

## Requirements

Please see `requirements.txt`.

## Quickstart

**TL;DR: run `prepare_data.sh`, then `main.py`.**

### Setting everything up

An entry point is the script `prepare_data.sh` that downloads the dataset and splits it into train/test 
parts in a 'stratified' manner, i.e. keeping all labels with just a single image in the training set, 
yet preserving similar label counts distributions in each part of the dataset. 

It should print

    DEBUG:root:Labels total: 172
    DEBUG:root:Labels seen just once: 37

before shutting down.

### Training

Training script `main.py` uses standard **hydra** configuration mechanism; the parameters one can modify
at the CLI call can be found in `configs/...`.

```bash
python3 main.py model.epochs=10
``` 

## How to cite

If you use the GlyphNet model, please cite the original work:

```bibtex
@article{barucci2021deep,
  title={A Deep Learning Approach to Ancient Egyptian Hieroglyphs Classification},
  author={Barucci, Andrea and Cucci, Costanza and Franci, Massimiliano and Loschiavo, Marco and Argenti, Fabrizio},
  journal={IEEE Access},
  volume={9},
  pages={123438--123447},
  year={2021},
  publisher={IEEE}
}
```

If you use the dataset, please cite the original work:

```bibtex
@inproceedings{franken2013automatic,
  title={Automatic Egyptian hieroglyph recognition by retrieving images as texts},
  author={Franken, Morris and van Gemert, Jan C},
  booktitle={Proceedings of the 21st ACM international conference on Multimedia},
  pages={765--768},
  year={2013}
}
```

Citing this repository is also appreciated:

```bibtex
@misc{glyphnetpytorch2021alekseev,
  title     = {{alexeyev/glyphnet-pytorch: GlyphNet, PyTorch implementation}},
  author    = {Anton Alekseev}, 
  year      = {2021},
  url       = {https://github.com/alexeyev/glyphnet-pytorch},
  language  = {english}
}
```

## TODO

* Add a practical usage scenario using [data augmentation](https://albumentations.ai/)
* Add an end-to-end image-to-prediction inference script using a pre-trained GlyphNet model 

## Notes

* [morrisfranken/glyphreader](https://github.com/morrisfranken/glyphreader), 
  the source of the [Pyramid of Unas](https://en.wikipedia.org/wiki/Pyramid_of_Unas) data
* Please do not confuse this work with another [GlyphNet project](https://github.com/noahtren/GlyphNet) 
  training networks to communicate using a visual language