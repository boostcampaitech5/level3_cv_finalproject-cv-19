import os

import torch


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.pt")
    # torch.jit.save(model, "temp.pt")
    size = os.path.getsize("temp.pt") / 1e6
    print(f"Model size: {size:.2f}MB")
    os.remove("temp.pt")
