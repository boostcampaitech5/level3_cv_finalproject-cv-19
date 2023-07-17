import torch

import clip

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32")

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# ['RN50', 'RN101', 'RN50x4', 'RN50x16','RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
# RN50     102,007,137  38,316,896
# RN101    119,688,033
# RN50x4   178,300,601
# ViT-B/32 151,277,313  87,849,216

# ViT-L/14 427,616,513
# ViT-L/14@336px 427,944,193


# from torchvision.models import resnet50

# myModel = resnet50(pretrained=True)
# print(sum(p.numel() for p in myModel.parameters() if p.requires_grad))
# 25,557,032
