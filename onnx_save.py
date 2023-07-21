# 1. Define a custom config

from typing import Dict

from optimum.exporters.onnx import export_models
from optimum.exporters.onnx.model_configs import CLIPTextOnnxConfig, ViTOnnxConfig
from transformers.models.clip import CLIPTextModelWithProjection, CLIPVisionModelWithProjection


class CLIPVisionOnnxConfig(ViTOnnxConfig):
    pass


class CLIPTextModelWithProjectionOnnxConfig(CLIPTextOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "text_embeds": {0: "batch_size"},
        }


class CLIPVisionModelWithProjectionOnnxConfig(CLIPVisionOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "image_embeds": {0: "batch_size"},
        }


# 2. Export to ONNX

text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

export_models(
    models_and_onnx_configs={
        "text_model": (text_model, CLIPTextModelWithProjectionOnnxConfig(text_model.config)),
        "vision_model": (vision_model, CLIPVisionModelWithProjectionOnnxConfig(vision_model.config)),
    },
    output_dir="models",
)
