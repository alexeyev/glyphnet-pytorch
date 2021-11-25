# glyphnet-pytorch
...a.k.a. cracking **MNIST: Ancient Egypt Edition**

This repository presents an implementation of the **Glyphnet** 
classifier introduced in the work [A Deep Learning Approach 
to Ancient Egyptian Hieroglyphs 
Classification](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9528382) 
and apply it to the data accompanying the work ["Automatic Egyptian 
Hieroglyph Recognition by Retrieving Images as Texts", 2013](...) 
(NB! Glyphnet paper uses a larger dataset).

We hope that this PyTorch implementation of the model will encourage
the further research in this direction; there is clearly a lot 
of work to do.

## Research Questions to Be Addressed in the Future

1. Is the Unas pyramids dataset as easy-to-solve as MNIST? Should check kNN/MLP approaches.
2. If not, the model can be informed and simplified using a pattern matching layer (unicode 
hieroglyphs scaled to the required image size).
3. Metric learning approach should also be worthwhile; judging by the data, mining hard negative
samples could help.

## Notes

* Please do not confuse this work with another [GlyphNet project](https://github.com/noahtren/GlyphNet) 
training networks to communicate using a visual language