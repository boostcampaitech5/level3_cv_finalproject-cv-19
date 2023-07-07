import os

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

BASE_PATH = "./weights"
SAVE_PATH = "./weights_save"

for i in os.listdir(BASE_PATH):
    file_name = os.path.join(BASE_PATH, i)
    save_name = os.path.join(SAVE_PATH, i)

    model = torch.load(file_name)
    model.eval()
    # print(model)

    scripted_model = torch.jit.script(model)
    optimized_model = optimize_for_mobile(scripted_model)
    optimized_model._save_for_lite_interpreter(f"{save_name[:-3]}.ptl")

    print("model successfully exported")
