import os
from glob import glob
from typing import List, Union
import json
import torch
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

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

def TextEncoder(text):
    device = "cpu"
    text = tokenize(text).to(device) # torch.Size([1, 77])
    text_model = torch.jit.load("motis_weight/final_text_encoder_4.pt", map_location="cpu").to(device).eval()  # Text Transformer
    with torch.no_grad():
        language_emb = text_model(text)  # torch.Size([1, 512])
    # print(json.dumps(language_emb.tolist()))
    return language_emb.tolist()