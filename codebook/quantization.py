import argparse

import torch
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection


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

    parser.add_argument("--model", type=str, default="clip", help="which model to use")
    parser.add_argument("--quantization_method", type=str, default=False, help="which method to use")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(args)

    return args


def main(args):
    # Load the huggingface CLIP model
    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(args.device).eval()
    text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(args.device).eval()

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
