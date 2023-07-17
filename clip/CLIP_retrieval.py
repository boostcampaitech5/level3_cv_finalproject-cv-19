import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import clip

"""
user input query에 대해서 결과를 검색하는 코드
DB는 MS-COCO validation set으로 구성, benchmarking과 마찬가지로 feature store를 뽑아 놓고 가져와서 사용
"""


def parse_args():
    parser = argparse.ArgumentParser()

    # Conventional args
    parser.add_argument("--model", type=str, default="clip", help="which model to use for feature extract")
    parser.add_argument("--data_path", type=str, default="/opt/ml/MS-COCO/val2017", help="data path to use")

    parser.add_argument("--batch_size", type=int, default=1, help="input batch size for validing (default: 1000)")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    print(args)

    return args


# Function that computes the feature vectors for a batch of images
def compute_clip_features(photos_batch, model, preprocess):
    # Load all the photos from the files
    photos = [Image.open(photo_file) for photo_file in photos_batch]

    # Preprocess all photos
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(args.device)

    with torch.no_grad():
        # Encode the photos batch to compute the feature vectors and normalize them
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)

    # Transfer the feature vectors back to the CPU and convert to numpy
    return photos_features.cpu().numpy()


def extract_coco_features(feature_path, model, preprocess):
    tmp_path = feature_path / "tmp"
    ensure_dir(feature_path)
    ensure_dir(tmp_path)

    photos_path = Path(args.data_path)
    photos_files = list(photos_path.glob("*.jpg"))  # List all JPGs in the folder
    print(f"Photos found: {len(photos_files)}")  # Print some statistics

    # Define the batch size so that it fits on your GPU. You can also do the processing on the CPU, but it will be slower.
    batch_size = 32

    # Compute how many batches are needed
    batches = math.ceil(len(photos_files) / batch_size)

    # Process each batch
    for i in range(batches):
        print(f"Processing batch {i+1}/{batches}")

        batch_ids_path = tmp_path / f"{i:010d}.csv"
        batch_features_path = tmp_path / f"{i:010d}.npy"

        # Only do the processing if the batch wasn't processed yet
        if not batch_features_path.exists():
            try:
                # Select the photos for the current batch
                batch_files = photos_files[i * batch_size : (i + 1) * batch_size]

                # Compute the features and save to a numpy file
                batch_features = compute_clip_features(batch_files, model, preprocess)
                np.save(batch_features_path, batch_features)

                # Save the photo IDs to a CSV file
                photo_ids = [photo_file.name.split(".")[0] for photo_file in batch_files]
                photo_ids_data = pd.DataFrame(photo_ids, columns=["photo_id"])
                photo_ids_data.to_csv(batch_ids_path, index=False)
            except:
                # Catch problems with the processing to make the process more robust
                print(f"Problem with batch {i}")

    # Load all numpy files
    features_list = [np.load(features_file) for features_file in sorted(tmp_path.glob("*.npy"))]

    # Concatenate the features and store in a merged file
    features = np.concatenate(features_list)
    np.save(feature_path / "features.npy", features)

    # Load all the photo IDs
    photo_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(tmp_path.glob("*.csv"))])
    photo_ids.to_csv(feature_path / "photo_ids.csv", index=False)

    os.system(f"rm -r {tmp_path}")


def main(args):
    # Load the open CLIP model
    device = args.device
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Load the photo IDs
    feature_path = Path("/opt/ml/level3_cv_finalproject-cv-19/COCO-benchmarking/feature_store") / args.model

    # Pre-compute image DB features
    if not feature_path.is_dir():
        extract_coco_features(feature_path, model, preprocess)

    photo_ids = pd.read_csv(os.path.join(feature_path, "photo_ids.csv"))
    photo_ids = list(photo_ids["photo_id"])

    # Load the features vectors
    photo_features = np.load(os.path.join(feature_path, "features.npy"))

    # Convert features to Tensors: Float32 on CPU and Float16 on GPU
    if device == "cpu":
        photo_features = torch.from_numpy(photo_features).float().to(device)
    else:
        photo_features = torch.from_numpy(photo_features).to(device)

    # Print some statistics
    print(f"Photos loaded: {len(photo_ids)}")

    dataset = MyDataset(args.label_path, args.data_path, preprocess)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=False, drop_last=False
    )

    print("\nCalculating inference results..")
    correct, total = 0, 0
    with torch.no_grad():
        for idx, (target, captions) in enumerate(tqdm(dataloader)):
            # Encode and normalize the search query using CLIP
            text_encoded = model.encode_text(clip.tokenize(captions[0]).to(device))  # torch.Size([5, 512])
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

            # Compute the similarity between the search query and each photo using the Cosine similarity
            similarities = (text_encoded @ photo_features.T).squeeze(1)  # torch.Size([5, 5000])

            # Sort the photos by their similarity score
            best_photo_idx = (-similarities).argsort()

            # Return the photo IDs of the best matches
            best_photo_ids = [[photo_ids[j] for j in i[: args.num_compare]] for i in best_photo_idx]  # [: args.num_compare]

            # best_photo_ids = [[photo_ids[j] for j in i] for i in best_photo_idx]
            # print(best_photo_ids[0].index(target[0]))

            id_exist = [1 if target[0] in i else 0 for i in best_photo_ids]
            correct += sum(id_exist)
            total += len(id_exist)

    accuracy = correct / total * 100
    print(f"Acc: {accuracy:.3f}%")


# python inference.py
if __name__ == "__main__":
    args = parse_args()
    main(args)
