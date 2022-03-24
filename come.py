import streamlit as st
import pandas as pd
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow

st.header('circle counter')
src = st.file_uploader('画像内の米粒をカウントして表示します。')
#1なら出来る
#src = 'out_sample2.jpg'
#src = '/content/drive/MyDrive/Colab Notebooks/making/rice_conunt/mugi.jpg'
#
def cell(src):
    img = cv2.imread(src)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    cv2.morphologyEx(bin_img,cv2.MORPH_OPEN,kernel,iterations=2)
    sure_bg = cv2.dilate(bin_img,kernel,iterations=4)
    dist = cv2.distanceTransform(bin_img,cv2.DIST_L2,5)
    ret ,sure_fg = cv2.threshold(dist,0.70*dist.max(),255,cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret,merkers = cv2.connectedComponents(sure_fg)
    merkers += 1
    merkers[unknown == 255] = 0
    # watershed アルゴリズムを適用する。
    merkers = cv2.watershed(img, merkers)
    labels = np.unique(merkers)

    coins = []
    for label in labels[2:]:  # 0:背景ラベル １：境界ラベル は無視する。

    # ラベル label の領域のみ前景、それ以外は背景となる2値画像を作成する。
        target = np.where(merkers == label, 255, 0).astype(np.uint8)
    # 作成した2値画像に対して、輪郭抽出を行う。
        contours, hierarchy = cv2.findContours(
        target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        coins.append(contours[0])
# 輪郭を描画する。
    cv2.drawContours(img, coins, -1, color=(0, 0, 255), thickness=2)
    st.image(img)
    con = len(coins)
    st.write(f'粒数は、{con}です。')

if src:
    cell()
else:
    st.write('こちらに表示予定')