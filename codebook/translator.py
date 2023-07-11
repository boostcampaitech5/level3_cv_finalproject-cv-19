import os
import sys
import urllib.request
import json
client_id = "4tyat0z92a"  # 네이버 클라우드 플랫폼 Papago API 클라이언트 ID
client_secret = "Dm2JKtF4u1jrTObtee4ES7j1U8RIkf70UzpRlzkJ"  # 네이버 클라우드 플랫폼 Papago API 클라이언트 시크릿

encText = urllib.parse.quote("번역할 문장을 입력하세요")
data = "source=ko&target=en&text=" + encText
url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
request = urllib.request.Request(url)
request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
request.add_header("X-NCP-APIGW-API-KEY",client_secret)
response = urllib.request.urlopen(request, data=data.encode("utf-8"))
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(json.loads(response_body.decode('utf-8'))["message"]["result"]["translatedText"])
else:
    print("Error Code:" + rescode)