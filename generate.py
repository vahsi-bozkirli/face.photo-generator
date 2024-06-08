#generate.py
#!/usr/bin/env/python3

"""
generates image with autoencoder model the given images
"""

import torch

def predict(model, data):
    with torch.no_grad():
        inputs = data.float()
        outputs = model(inputs)
    return outputs
