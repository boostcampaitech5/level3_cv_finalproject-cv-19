import os
from glob import glob

import torch
from PIL import Image

import clip

# model, 전처리 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# 폴더 내의 jpg 이미지 파일들을 찾기 위한 경로 패턴
img_path = os.path.join("/opt/ml/level3_cv_finalproject-cv-19/data", "*.jpg")

# 폴더 내의 jpg 이미지 파일들을 리스트로 가져오기
image_files = glob(img_path)

# 사용할 태그 목록
tags = ["animal", "food", "human", "indoor", "outdoor"]
texts = [f"A photo of a {tag}" for tag in tags]

# 한번의 for loop에서 하나의 이미지, 텍스트 데이터에 대해서 유사도 및 확률값을 구합니다.
for idx, path in enumerate(image_files):
    img = Image.open(path)

    # plt.figure(figsize=(4, 4))
    # plt.title(f"Test Image {idx+1}")
    # plt.imshow(img)
    # plt.show()

    # 이미지와 텍스트를 벡터로 만들고 유사도와 확률값을 구합니다.
    image = preprocess(img).unsqueeze(0).to(device)
    text_dataset = texts
    text = clip.tokenize(text_dataset).to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy().flatten()

    print(f"\nName of image: {path.split('/')[-1]}")
    print("- Similarity between Text and Image -")
    for i in range(len(text_dataset)):
        print("  " + text_dataset[i] + ":", logits_per_image.cpu().numpy().flatten()[i])

    print("\n- The probability that each text matches the image -")
    for i in range(len(text_dataset)):
        print("  " + text_dataset[i] + ":", round(probs[i] * 100, 3), "%")
