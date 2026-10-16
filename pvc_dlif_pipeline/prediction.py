"""Predicting an input function with a trained network."""

import numpy as np
import torch

from pvc_dlif_pipeline.training import network_prediction


def load_weights(network, model_path, device):
    """Load trained weights into a network and put it in evaluation mode.

    model.pt holds the state dict; the original pipeline stored it under the
    key "state_dict", so both forms are accepted.
    """
    saved = torch.load(str(model_path), map_location=device, weights_only=False)
    if isinstance(saved, dict) and "state_dict" in saved:
        saved = saved["state_dict"]
    network.load_state_dict(saved)
    network.to(device)
    network.eval()
    return network


def predict_curve(network, image, device):
    """Predict one input function (one value per frame) from an image (frames, z, y, x).

    Prediction runs in float32, without mixed precision.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)[np.newaxis, np.newaxis]).to(device)
    with torch.no_grad():
        prediction = network_prediction(network, tensor)
    return prediction.detach().cpu().numpy().reshape(-1).astype(np.float64)
