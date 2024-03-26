"""
ターミナル上で「streamlit run image_uploader_3.py」と入力し起動します。
"""

import pandas as pd
import numpy as np
from PIL import Image
import requests
import streamlit as st
import cv2
from streamlit_drawable_canvas import st_canvas
from io import BytesIO
import io

#############################################################前準備#############################################################
# データフレームの作成
names = ['りんご', 'ねこ', 'いぬ', 'いるか', 'きりん', 'はむすたー', 'ぺんぎん', 'うさぎ', 'いちご', 'とんかつ']
id = np.arange(0, len(names))
df_name = pd.DataFrame({'id': id, 'name': names})

hiragana = ['あ', 'い', 'う', 'え', 'お',
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
id = np.arange(0, len(hiragana))
df_hiragana = pd.DataFrame({'id': id, 'hiragana': hiragana})





# 入力文字の取得------------------------------------------------------------------------------------------------------------------
def get_text(canvas):
    #st.image(canvas.image_data,'前')
    word = ''
    # canvas画像(ひらがな)をbytes型に変換
    canvas_image = Image.fromarray(canvas.image_data.astype('uint8'))

    image_bytes = io.BytesIO()
    canvas_image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    api_url_hiragana = "http://127.0.0.1:8000/hiragana_classifier"

    #st.image(Image.open(io.BytesIO(image_bytes)).convert('RGB'),'あと')
    #st.image(Image.open(BytesIO(image_bytes)).convert("RGB"), '後')

    word = requests.post(api_url_hiragana, files={"file": ("image.jpg", image_bytes, "image/jpeg")})
    word = word.content.decode('utf-8')
    word = word.replace('"', '')
    #print(word)
    return word
##############################################################################################################################

# streamlit メイン処理コード
st.title('ひらがなれんしゅう')
st.write('入力できる画像')
st.table(df_name['name'])
uploaded_file = st.file_uploader('画像ファイルを選択してください', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # FastAPI url
    api_url = "http://127.0.0.1:8000/image_classifier"
    #with open(uploaded_file, 'rb') as image_file:

    # 画像をbytes型に変換
    image_up = uploaded_file.read()
    # print(image_data)
    # POSTリクエストを作成してAPIに送信
    response = requests.post(api_url, files={"file": ("image.jpg", image_up, "image/jpeg")})

    # print(response.content.decode('utf-8'))
    #Image_res = Image.open(uploaded_file)
    st.image(Image.open(uploaded_file))
    label = response.content.decode('utf-8')
    #labelの文字数を取得
    label = label.replace('"', '')
    # 列を作成
    l = len(label)
    columns = st.columns(l)
    canvas = []
    # 各列にst_canvasを配置
    for i, col in enumerate(columns):
        with columns[i]:
            #st.write(f"Canvas {i}")
            can =  st_canvas(
                fill_color="rgba(0, 0, 0, 0.3)",  # 色を設定
                stroke_width=10,  # 筆圧調整
                stroke_color="rgb(0, 0, 0)",  # ペンの色を設定
                background_color="#FFF",
                width=180,
                height=180,
                drawing_mode="freedraw",
                key=f"canvas{i}",)
            canvas.append(can)


    # 手書き文字を取得
    handwritten_texts = []

    if st.button("かけた"):
        for i in range(len(label)):
            with columns[i]:
                handwritten_text = get_text(canvas[i])
                
                print(handwritten_text)
                handwritten_texts.append(handwritten_text)
            print(handwritten_texts)
        answer = ""
        for i in range(len(handwritten_texts)):
            answer = answer+handwritten_texts[i]
    

    # 正解と突き合わせる
    if len(handwritten_texts) > 0:
        st.success(f"かいたのは: {answer}")
        #if st.button("クリア"):
        #    handwritten_text = [""]
    # 結果を表示
        #ans = str(' '.join(handwritten_texts))
        # st.write(answer, label)
        if answer == str(label):
            st.write("せいかい！！")
        else:
            st.write("ちがうよ")





