import argparse
import json

import numpy as np
import torch
from dataset import ImageNetDataset
from tqdm import tqdm

import clip

"""
COCO-caption text를 가지고 전체 val image와의 유사도를 계산했을 때, top-10 acc를 측정
MoTis, CLIP 등 모델 별 성능 benchmarking
"""


def parse_args():
    parser = argparse.ArgumentParser()

    # Conventional args
    parser.add_argument("--model", type=str, default="clip", help="which model to use for feature extract")
    parser.add_argument("--data_path", type=str, default="/opt/ml/ImageNet/val", help="data path to use")
    parser.add_argument("--label_path", type=str, default="/opt/ml/ImageNet/class_info.json", help="label path to use")
    # eng_re_translated_captions_val2017 eng_re_translated_captions_val2017_3 captions_val2017
    parser.add_argument("--batch_size", type=int, default=32, help="input batch size for validing (default: 1000)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    print(args)

    return args


def main(args):
    # Load the open CLIP model
    device = args.device
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    dataset = ImageNetDataset(args.label_path, args.data_path, preprocess)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=False, drop_last=False
    )

    with open("/opt/ml/ImageNet/class_info.json") as f:
        label = json.load(f)

    classes = [i["synset"][0] for i in label]
    text_prompt = [f"A photo of a {i}" for i in classes]

    print("\nCalculating inference results..")
    correct, total = 0, 0
    with torch.no_grad():
        # Encode and normalize the search query using CLIP
        print("text encoding ...")
        text_encoded = model.encode_text(clip.tokenize(text_prompt).to(device))  # torch.Size([1000, 512])
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

        print("image inference ...")
        for idx, (img, cls) in enumerate(tqdm(dataloader)):
            # Encode the photos batch to compute the feature vectors and normalize them
            photo_features = model.encode_image(torch.tensor(np.stack(img)).to(device))  # torch.Size([bs, 512])
            photo_features /= photo_features.norm(dim=-1, keepdim=True)

            # Compute the similarity between the search query and each photo using the Cosine similarity
            similarities = (photo_features @ text_encoded.T).squeeze(1)  # torch.Size([bs, 1000])

            # Sort the photos by their similarity score
            outputs = (-similarities).argsort()[:, 0]

            cor = [1 if i == j else 0 for i, j in zip(outputs, cls)]
            correct += sum(cor)
            total += len(cor)

    accuracy = correct / total * 100
    print(f"Acc: {accuracy:.3f}%")


# python inference.py
if __name__ == "__main__":
    args = parse_args()
    main(args)
