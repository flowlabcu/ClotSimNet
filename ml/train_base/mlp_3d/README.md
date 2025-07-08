# MLP 3D Scripts

Josh Gregory


Last edited: June 27, 2025

## Description

Each of these scripts trains an MLP variant (full image features, first-order image features only, dendrogram-based image features) on 3D CFD clot-pore simulation data. Any of the files that end in `_pretrain` utilize the pretrained MLP that was trained on 2D data and freezes every layer in that network **except for the final layer**, and retrains it on the 3D data. Five-fold cross-validation is used due to the small amount of data, and then the model is re-trained on the entire dataset.

Files ending in `_scratch` do the same thing as `_pretrain`, except they train the model from scratch and do not use any pretrained MLP models (e.g. train each model from scratch).