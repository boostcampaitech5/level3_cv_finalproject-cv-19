import requests
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

import clip


def preprocess(image, size=224):
    transform = Compose(
        [Resize(size), CenterCrop(size), ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]
    )
    return transform(image)


device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- image encoding -----------------
visual_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processed_img = preprocess(image).unsqueeze(0)
out_img = visual_model(processed_img)

image_embeds = out_img.image_embeds

# Creating the trace
traced_visual_model = torch.jit.trace(visual_model.to(device), processed_img.to(device), strict=False)
torch.jit.save(traced_visual_model, "clip_visual.pt")


# ----------------- text encoding -----------------
text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

texts = ["a photo of a cat", "a photo of a dog"]
texts = clip.tokenize(texts)  # tokenize

out_text = text_model(texts)
text_embeds = out_text.text_embeds

# Creating the trace
traced_text_model = torch.jit.trace(text_model.to(device), texts.to(device), strict=False)
torch.jit.save(traced_text_model, "clip_text.pt")

print("image_embeds", image_embeds.shape)
print("text_embeds", text_embeds.shape)


from torch.utils.mobile_optimizer import optimize_for_mobile

visual_optimized_model = optimize_for_mobile(traced_visual_model)
visual_optimized_model._save_for_lite_interpreter("clip_visual_mobile.ptl")

text_optimized_model = optimize_for_mobile(traced_text_model)
text_optimized_model._save_for_lite_interpreter("clip_text_mobile.ptl")
