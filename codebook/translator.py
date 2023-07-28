import json
import os
import sys
import urllib.request

client_id = "CLIENT_ID"  # CLIENT_ID 자리에 할당받은 네이버 클라우드 플랫폼 Papago API 클라이언트 아이디를 입력
client_secret = "CLIENT_SECRET"  # CLIENT_SECRET 자리에 할당받은 네이버 클라우드 플랫폼 Papago API 클라이언트 시크릿을 입력

encText = urllib.parse.quote("번역할 문장을 입력하세요")
data = "source=ko&target=en&text=" + encText
url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
request = urllib.request.Request(url)
request.add_header("X-NCP-APIGW-API-KEY-ID", client_id)
request.add_header("X-NCP-APIGW-API-KEY", client_secret)
response = urllib.request.urlopen(request, data=data.encode("utf-8"))
rescode = response.getcode()
if rescode == 200:
    response_body = response.read()
    print(json.loads(response_body.decode("utf-8"))["message"]["result"]["translatedText"])
else:
    print("Error Code:" + rescode)
