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

# print(vision_model)

# CLIPVisionModelWithProjection(
#   (vision_model): CLIPVisionTransformer(
#     (embeddings): CLIPVisionEmbeddings(
#       (patch_embedding): Conv2d(3, 768, kernel_size=(32, 32), stride=(32, 32), bias=False)
#       (position_embedding): Embedding(50, 768)
#     )
#     (pre_layrnorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#     (encoder): CLIPEncoder(
#       (layers): ModuleList(
#         (0-11): 12 x CLIPEncoderLayer(
#           (self_attn): CLIPAttention(
#             (k_proj): Linear(in_features=768, out_features=768, bias=True)
#             (v_proj): Linear(in_features=768, out_features=768, bias=True)
#             (q_proj): Linear(in_features=768, out_features=768, bias=True)
#             (out_proj): Linear(in_features=768, out_features=768, bias=True)
#           )
#           (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#           (mlp): CLIPMLP(
#             (activation_fn): QuickGELUActivation()
#             (fc1): Linear(in_features=768, out_features=3072, bias=True)
#             (fc2): Linear(in_features=3072, out_features=768, bias=True)
#           )
#           (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#     )
#     (post_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#   )
#   (visual_projection): Linear(in_features=768, out_features=512, bias=False)
# )
