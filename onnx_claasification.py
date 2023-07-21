import onnxruntime
import requests
from PIL import Image
from transformers import AutoProcessor

# Load the ONNX model
vision_model = onnxruntime.InferenceSession("models/vision_model.onnx")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Set the input data
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processed_img = processor(images=image, return_tensors="pt")
input_data = {"pixel_values": processed_img["pixel_values"].cpu().numpy()}

# Run the model
output_data = vision_model.run(None, input_data)

# Print the output
print(output_data.shape)
