from icrawler.builtin import GoogleImageCrawler
import sys
import cv2
import os

# echo
def echo(text):
    print("--->" + text)

# imageで指定した画像をrectの範囲でクロップして返す
def crop(image, rect):
    x, y, width, height = rect
    return image[y:y + height, x:x + width]


# rectで指定した矩形の左上の座標を返す
def top_left(rect):
    x, y, _, _ = rect
    return (x, y)


# rectで指定した矩形の右下の座標を返す
def bottom_right(rect):
    x, y, width, height = rect
    return (x + width, y + height)


# imageで指定した画像に指定したcolorのrectの矩形を描画して返す
def drawRect(image, rect, color):
    return cv2.rectangle(
        image,
        top_left(rect),
        bottom_right(rect),
        color,
        3)


# 指定したキーワードの画像を取得して顔領域を切り出して保存
def fetchAndCropFace(keyword, max_num):
    echo("fetchAndCropFace")
    input_file_path = f"./training_data/original/{keyword}/"
    output_file_path = f"./training_data/cropped_face/{keyword}/"

    crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=4,
        storage={"root_dir": input_file_path})
    crawler.session.verify = False

    crawler.crawl(keyword=keyword, max_num=int(max_num))

    echo("入力ファイルパス: " + input_file_path)
    os.makedirs(input_file_path, exist_ok=True)
    echo("出力ファイルパス: " + output_file_path)
    os.makedirs(output_file_path, exist_ok=True)
    input_files = os.listdir(input_file_path)

    echo("画像取得枚数: " + str(len(input_files)))
    echo("顔認識用特徴量ファイルを読み込み") #--- （カスケードファイルのパスを指定）
    cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

    echo("認識結果表示用のwindowを作成")
    windowName = 'window'
    cv2.namedWindow(windowName, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 500, 500)

    for input_file in input_files:
        input_image = cv2.imread(input_file_path + input_file)

        if input_image is None:
            continue

        height, width, _ = input_image.shape
        cv2.imshow(windowName, input_image)
        cv2.waitKey(50)

        echo("顔認識の実行")
        face_rects = cascade.detectMultiScale(
            input_image,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(10, 10))

        if len(face_rects) == 0:
            echo("顔の検出に失敗")
            continue

        echo("顔の検出に成功")

        echo("最初に検出した顔のみ取得")
        face_rect = face_rects[0]

        echo("顔領域だけ切り出して保存")
        output_image = crop(input_image, face_rect)
        cv2.imwrite(output_file_path + input_file, output_image)

        echo("顔領域に矩形を描画して表示")
        marked_input_image = drawRect(input_image, face_rect, (0, 255, 0))
        cv2.imshow(windowName, marked_input_image)
        cv2.waitKey(50)

    # windowを破棄
    cv2.destroyAllWindows()

# メイン関数
def main():
    max_num = sys.argv[1]
    keywords = sys.argv[2:len(sys.argv)]

    for keyword in keywords:
        fetchAndCropFace(keyword, max_num)


main()
