import os
from glob import glob
from typing import List, Union

import torch
from PIL import Image
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


def compare_embeddings(logit_scale, img_embs, txt_embs):
    # normalized features
    image_features = img_embs / img_embs.norm(dim=-1, keepdim=True)
    text_features = txt_embs / txt_embs.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text


def preprocess(image, size=224):
    transform = Compose(
        [Resize(size), CenterCrop(size), ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]
    )
    return transform(image)


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if torch.__version__ < "1.8.0":
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, : len(tokens)] = torch.tensor(tokens)

    return result


# model, 전처리 로드
device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"


# 이미지 인코더 생성
image_resolution = 224  # 이미지 해상도
vision_model = torch.jit.load("../motis_weight/final_visual.pt").to(device).eval()  # Vision Transformer
# model.visual = vision_model

# 텍스트 인코더 생성
text_model = torch.jit.load("../motis_weight/final_text_encoder_4.pt").to(device).eval()  # Text Transformer
# model.transformer = text_model

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

    # 이미지와 텍스트를 벡터로 만들고 유사도와 확률값을 구합니다.
    image = preprocess(img).unsqueeze(0).to(device)  # torch.Size([1, 3, 224, 224])
    text_dataset = texts
    text = tokenize(text_dataset).to(device)
    with torch.no_grad():
        image_embs = vision_model(image).float().to("cpu")  # torch.Size([1, 512])
        language_embs = text_model(text)  # torch.Size([5, 512])

        similarity = image_embs @ language_embs.T
        probs = similarity.softmax(dim=-1).cpu().numpy().flatten()

        # CLIP Temperature scaler
        # logit_scale = (torch.ones([]) * np.log(1 / 0.07)).exp().float().to("cpu")
        # logit_scale = torch.ones([]).exp().float().to("cpu")
        # img_logits, txt_logits = compare_embeddings(logit_scale, image_embs, language_embs)

        # probs = img_logits.softmax(dim=-1).cpu().detach().numpy()
        # probs = (np.around(probs, decimals=5) * 100).item()
        # print(probs)

    print(f"\nName of image: {path.split('/')[-1]}")
    print("- Similarity between Text and Image -")
    for i in range(len(text_dataset)):
        print("  " + text_dataset[i] + ":", similarity.numpy().flatten()[i])

    print("\n- The probability that each text matches the image -")
    for i in range(len(text_dataset)):
        print("  " + text_dataset[i] + ":", round(probs[i] * 100, 3), "%")
