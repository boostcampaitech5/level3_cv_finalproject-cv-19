import argparse

import torch
from constants import imagenet_classes, imagenet_templates
from imagenetv2_pytorch import ImageNetV2Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

import clip


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    # Conventional args
    # openai-clip ['RN50', 'RN101', 'RN50x4', 'RN50x16','RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    # huggingface-clip ./weights/clip/traced_visual_model.pt ./weights/clip/traced_text_model.pt
    # motis ./weights/motis/final_visual.pt ./weights/motis/final_text_encoder_4.pt
    parser.add_argument("--model", type=str, default="openai-clip", help="which model to use for feature extract")
    parser.add_argument("--load_ckpt", type=str2bool, default=False, help="determine whether load model from checkpoint")

    parser.add_argument("--visual_path", type=str, default=None, help="visual model weight path")
    parser.add_argument("--text_path", type=str, default=None, help="text model weight path")
    parser.add_argument("--batch_size", type=int, default=32, help="input batch size for validing (default: 1000)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(args)

    return args


def preprocess(image, size=224):
    transform = Compose(
        [Resize(size), CenterCrop(size), ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]
    )
    return transform(image)


def zeroshot_classifier(classnames, templates, text_model, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).to(args.device)  # tokenize

            # tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            # texts = tokenizer(texts, padding=True, return_tensors="pt").to(args.device)

            if args.model == "openai-clip":
                class_embeddings = text_model.encode_text(texts)  # embed with text encoder
            elif args.model == "huggingface-clip":
                class_embeddings = text_model(texts)
                class_embeddings = class_embeddings["text_embeds"]
            else:
                class_embeddings = text_model(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def main(args):
    # Load the open CLIP model
    # ['RN50', 'RN101', 'RN50x4', 'RN50x16','RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    if args.model == "openai-clip":
        model, _ = clip.load("RN50")
        model.to(args.device).eval()

    elif args.model == "huggingface-clip":
        if args.load_ckpt:
            vision_model = torch.jit.load(args.visual_path).to(args.device).eval()  # Vision Transformer
            text_model = torch.jit.load(args.text_path).to(args.device).eval()  # Text Transformer
        else:
            vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(args.device).eval()
            text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(args.device).eval()

    elif args.model == "motis":
        args.device = "cpu"
        vision_model = torch.jit.load(args.visual_path, map_location=torch.device(args.device)).eval()  # Vision Transformer
        text_model = torch.jit.load(args.text_path, map_location=torch.device(args.device)).eval()  # Text Transformer

    images = ImageNetV2Dataset(transform=preprocess)
    loader = torch.utils.data.DataLoader(images, batch_size=args.batch_size, num_workers=4)

    if args.model == "openai-clip":
        zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates, model, args)
    else:
        zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates, text_model, args)

    print("\nCalculating inference results..")
    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.to(args.device)
            target = target.to(args.device)

            # predict
            if args.model == "openai-clip":
                image_features = model.encode_image(images)
            elif args.model == "huggingface-clip":
                image_features = vision_model(images)
                image_features = image_features["image_embeds"]
            else:
                image_features = vision_model(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ zeroshot_weights

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")


# python inference.py
if __name__ == "__main__":
    args = parse_args()
    main(args)
