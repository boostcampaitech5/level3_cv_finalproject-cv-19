import os

import requests
import torch
from PIL import Image
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

BASE_PATH = "/opt/ml/level3_cv_finalproject-cv-19/clip_compression/weight"


def load_model(model_file):
    model = torch.jit.load(os.path.join(BASE_PATH, model_file))
    # model.to("cpu")
    return model


def preprocess(image, size=224):
    transform = Compose(
        [Resize(size), CenterCrop(size), ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]
    )
    return transform(image)


def save_model(visual_model, quant_type, device):
    # ----------------- image encoding -----------------
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    processed_img = preprocess(image).unsqueeze(0)
    if quant_type == "fp16":
        processed_img = processed_img.half()

    # Creating the trace
    traced_visual_model = torch.jit.trace(visual_model.to(device), processed_img.to(device), strict=False)
    torch.jit.save(traced_visual_model, f"./weight/visual_{quant_type}_{device}.pt")

    visual_optimized_model = optimize_for_mobile(traced_visual_model)
    visual_optimized_model._save_for_lite_interpreter(f"./weight/visual_mobile_{quant_type}_{device}.ptl")

    # # ----------------- text encoding -----------------
    # texts = ["a photo of a cat", "a photo of a dog"]
    # texts = clip.tokenize(texts)  # tokenize

    # # Creating the trace
    # traced_text_model = torch.jit.trace(text_model.to(device), texts.to(device), strict=False)
    # torch.jit.save(traced_text_model, f"./weight/text_{quant_type}.pt")

    # text_optimized_model = optimize_for_mobile(traced_text_model)
    # text_optimized_model._save_for_lite_interpreter(f"./weight/text_mobile_{quant_type}.ptl")


def bit_save_model(visual_model, quant_type):
    # ----------------- image encoding -----------------
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    processed_img = preprocess(image).unsqueeze(0).half()

    # Creating the trace
    traced_visual_model = torch.jit.trace(visual_model, processed_img, strict=False)
    torch.jit.save(traced_visual_model, f"./weight/visual_{quant_type}.pt")

    # visual_optimized_model = optimize_for_mobile(traced_visual_model)
    # visual_optimized_model._save_for_lite_interpreter(f"./weight/visual_mobile_{quant_type}.ptl")


if __name__ == "__main__":
    from transformers import CLIPVisionModelWithProjection

    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

    # Test fp16 model
    fp16_model = model.half()

    save_model(fp16_model, "fp16", "cuda")
