import os
import torch
from src.models.spatial_baseline import SpatialBaselineCNN


def load_baseline_model(device):

    local_model_path = os.path.join("models", "baseline_model.pth")
    kaggle_model_path = "/kaggle/working/baseline_model.pth"

    if os.path.exists(local_model_path):
        model_path = local_model_path
        print("Running locally.")
    else:
        model_path = kaggle_model_path
        print("Running in Kaggle.")

    model = SpatialBaselineCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model