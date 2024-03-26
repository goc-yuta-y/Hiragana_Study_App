"""
ターミナルで「uvicorn image_classifier_main_3:app --reload」と入力して起動し、
別途image_uploader_3.pyを実行することで、streamlit上で画像をアップロードすると
画像分類結果が表示されます。
"""

from fastapi import FastAPI, File, UploadFile, Request
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
import pytorch_lightning as pl
import streamlit as st
from io import BytesIO
import numpy as np
import pandas as pd
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


app = FastAPI()

"""
# CORSミドルウェアの有効化
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 全てのオリジンを許可
    allow_credentials=True,
    allow_methods=["*"],  # 全てのHTTPメソッドを許可
    allow_headers=["*"],  # 全てのヘッダーを許可
)
"""

# ResNet を特徴抽出機として使用
feature = resnet18(pretrained=True)

# ネットワークの定義
## 画像分類モデル用
class Net_image(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = resnet18(pretrained=True)
        self.fc = torch.nn.Linear(1000, 10)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h

## ひらがな分類モデル用
class Net_hiragana(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = resnet18(pretrained=True)
        self.fc = torch.nn.Linear(1000, 72)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h



# 画像分類モデル(model_image.pt)のロード
net = Net_image().cpu().eval()
net.load_state_dict(torch.load('model_image.pt', map_location=torch.device('cpu')))

# データフレーム(画像の種類)の作成
names = ['りんご', 'ねこ', 'いぬ', 'いるか', 'きりん', 'はむすたー', 'ぺんぎん', 'うさぎ', 'いちご', 'とんかつ']
id = np.arange(0, len(names))
df_name = pd.DataFrame({'id': id, 'name': names})

# ひらがな分類モデル(model_hiragana.pt)のロード
net_2 = Net_hiragana().cpu().eval()
net_2.load_state_dict(torch.load('model_hiragana_3.pt', map_location=torch.device('cpu')))

# データフレーム(ひらがな)の作成
charas = ['あ', 'い', 'う', 'え', 'お',
         'か', 'き', 'く', 'け', 'こ',
         'さ', 'し', 'す', 'せ', 'そ',
         'た', 'ち', 'つ', 'て', 'と',
         'な', 'に', 'ぬ', 'ね', 'の',
         'は', 'ひ', 'ふ', 'へ', 'ほ',
         'ま', 'み', 'む', 'め', 'も',
         'や', 'ゆ', 'よ',
         'ら', 'り', 'る', 'れ', 'ろ',
         'わ', 'を', 'ん',
         'が','ぎ','ぐ','げ','ご',
         'ざ','じ','ず','ぜ','ぞ',
         'だ','ぢ','づ','で','ど',
         'ば','び','ぶ','べ','ぼ',
         'ぱ','ぴ','ぷ','ぺ','ぽ','ー']
id_2 = np.arange(0, len(charas))
df_chara = pd.DataFrame({'id': id_2, 'chara': charas})

# トップページ
@app.get("/")
async def index():
    return{'class' : 'image_classifier'}

# メイン関数
@app.post("/image_classifier")
async def predict(file: bytes = File(...)):  # bytes型で受け取る
    # contents = await file.read()
    image = Image.open(BytesIO(file)).convert("RGB")

    # resnet18指定の画像前処理
    transform = transforms.Compose([
    transforms.Resize(224), # 画像の端が切れないように256にはしない
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)

    # 推論
    output = net(image.unsqueeze(0))

    # 予測結果の取得
    pred = output.argmax(dim=-1)
    label = ''
    # 予測結果から名前を取得
    label = df_name.iloc[int(pred), 1]
    # print('label', label)

    return label

@app.post("/hiragana_classifier")
async def predict(file: bytes = File(...)):  # bytes型で受け取る
    # contents = await file.read()
    image_2 = Image.open(BytesIO(file)).convert("RGB")

    # resnet18指定の画像前処理
    transform = transforms.Compose([
    transforms.Resize(224), #ひらがなの端が切れないように256にはしない
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_2 = transform(image_2)


    # 推論
    output_2 = net_2(image_2.unsqueeze(0))

    # 予測結果の取得
    pred_2 = output_2.argmax(dim=-1)

    # 予測結果から名前を取得
    # print(pred_2)
    label_2 = df_chara.iloc[int(pred_2), 1]
    # print('pred_2', pred_2)

    return label_2


# if __name__ == "__main__":
    # アプリの起動
    # uvicorn.run("classifier:app")
