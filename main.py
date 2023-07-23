from fastapi import FastAPI
# from starlette.responses import FileResponse
from fastapi.responses import FileResponse

import os
from glob import glob
from typing import List, Union

import torch
from CLIP.text_encoder import TextEncoder
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from dotenv import load_dotenv

import urllib.request
import json

app = FastAPI()

PTL_DIR = './models/'

load_dotenv()
client_id = os.environ.get("CLIENT_ID")
client_secret = os.environ.get("CLIENT_SECRET")


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/favicon.ico')
def favicon():
    file_path = os.path.join(app.root_path, "static/favicon.ico")
    return FileResponse(file_path)

@app.get("/tokenize/{text}")
def text_encoder(text : str):
    # text 번역
    encText = urllib.parse.quote(text)
    data = "source=ko&target=en&text=" + encText
    url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
    request.add_header("X-NCP-APIGW-API-KEY",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        translated_text = json.loads(response_body.decode('utf-8'))["message"]["result"]["translatedText"]
    else:
        print("Error Code:" + rescode)
    if translated_text:
        _text_encoder = TextEncoder(translated_text)
    else:
        _text_encoder = TextEncoder(text)
    return _text_encoder

@app.get("/{filename}")
def read_item(filename):
    targetFile = PTL_DIR + filename
    print(f"File Download : {targetFile}")
    return FileResponse(targetFile, filename=filename)
