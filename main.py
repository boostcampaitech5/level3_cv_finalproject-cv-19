from fastapi import FastAPI
# from starlette.responses import FileResponse
from fastapi.responses import FileResponse

import os
from glob import glob
from typing import List, Union

import torch
from clip.text_encoder import TextEncoder
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor




app = FastAPI()

PTL_DIR = './models/'

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/favicon.ico')
async def favicon():
    file_path = os.path.join(app.root_path, "static/favicon.ico")
    return FileResponse(file_path)

@app.get("/tokenize/{text}")
def text_encoder(text : str):
    _text_encoder = TextEncoder(text)
    return _text_encoder

@app.get("/{filename}")
def read_item(filename):
    targetFile = PTL_DIR + filename
    print(f"File Download : {targetFile}")
    return FileResponse(targetFile, filename=filename)
