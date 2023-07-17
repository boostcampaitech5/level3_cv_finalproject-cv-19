import requests
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection

# image encoding
visual_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processed_img = processor(images=image, return_tensors="pt")

out_img = visual_model(**processed_img)
image_embeds = out_img.image_embeds
# print("image_embeds", image_embeds.shape)  # (1, 512)

# text encoding
text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

texts = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

out_text = text_model(**texts)
text_embeds = out_text.text_embeds
print()
ex_out = text_model()
print(out_text.keys())
# print("text_embeds", text_embeds.shape)  # (2, 512)

# print(processed_img.keys())
# print(texts.keys())
# m = torch.jit.script(model)

# Creating the trace
# traced_model = torch.jit.trace(model, inputs["pixel_values"], strict=False)
# torch.jit.save(traced_model, "traced_clip_visual.pt")
# print(traced_model)
