import torch
from imagenetv2_pytorch import ImageNetV2Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm


def preprocess(image, size=224):
    transform = Compose(
        [Resize(size), CenterCrop(size), ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]
    )
    return transform(image)


def calibrate(model, device, time=False):
    max_idx = 3 if time else 30
    images = ImageNetV2Dataset(transform=preprocess)
    data_loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=4)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, (image, _) in enumerate(tqdm(data_loader)):
            image = image.to(device)
            model(image)

            if idx == max_idx:
                break


def bit_calibrate(model, time=False):
    max_idx = 3 if time else 30
    images = ImageNetV2Dataset(transform=preprocess)
    data_loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=4)

    model.eval()
    with torch.no_grad():
        for idx, (image, _) in enumerate(tqdm(data_loader)):
            image = image.half()
            model(image)

            if idx == max_idx:
                break


if __name__ == "__main__":
    from transformers import CLIPVisionModelWithProjection

    visual_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    model_dynamic_quantized = torch.quantization.quantize_dynamic(visual_model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

    calibrate(model_dynamic_quantized, "cpu")
