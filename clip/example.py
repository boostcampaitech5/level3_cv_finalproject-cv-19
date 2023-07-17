import torch
from torchsummary import summary

import clip

device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

vision_model = torch.jit.load("./motis_weight/final_visual.pt").to(device)  # Vision Transformer
text_model = torch.jit.load("./motis_weight/final_text_encoder_4.pt").to(device)  # Text Transformer

summary(vision_model, (3, 224, 224), batch_size=1)

# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print(params)

# vision_parameters = filter(lambda p: p.requires_grad, vision_model.parameters())
# params = sum([np.prod(p.size()) for p in vision_parameters])
# print(params)

# text_parameters = filter(lambda p: p.requires_grad, text_model.parameters())
# params = sum([np.prod(p.size()) for p in text_parameters])
# print(params)
