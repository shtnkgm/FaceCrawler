from icrawler.builtin import GoogleImageCrawler
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from PIL import Image

args = sys.argv

keyword = args[1]
max_num = args[2]

input_file_path = f"./training_data/original/{keyword}/"
output_file_path = f"./training_data/cropped_face/{keyword}/"

crawler = GoogleImageCrawler(storage={"root_dir": input_file_path})
crawler.crawl(keyword=keyword, max_num=int(max_num))

os.makedirs(output_file_path, exist_ok=True)

def get_file(dir_path):
    filenames = os.listdir(dir_path)
    return filenames

input_files = get_file(input_file_path)

for input_file in input_files:
    # 画像の読み込み
    input_image = cv2.imread(input_file_path + input_file)
    cv2.imshow('input_image',input_image)
    cv2.waitKey(100)

    # 顔認識用特徴量ファイルを読み込む --- （カスケードファイルのパスを指定）
    cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

    # 顔認識の実行
    face_rects = cascade.detectMultiScale(input_image,scaleFactor=1.1,minNeighbors=10,minSize=(10,10))

    # 顔の検出に失敗した場合
    if len(face_rects) == 0:
        continue

    # 最初に検出した顔のみ取得
    face_rect = face_rects[0]
    # 顔だけ切り出して保存
    x = face_rect[0]
    y = face_rect[1]
    width = face_rect[2]
    height = face_rect[3]
    output_image = input_image[y:y + height, x:x + width]
    cv2.imshow('input_image',output_image)
    cv2.waitKey(100)
    save_path = output_file_path + input_file

    #認識結果の保存
    cv2.imwrite(save_path, output_image)
    cv2.destroyAllWindows()
