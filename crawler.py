from icrawler.builtin import GoogleImageCrawler
import sys
import cv2
import os


# imageで指定した画像をrectの範囲でクロップして返す
def crop(image, rect):
    x = rect[0]
    y = rect[1]
    width = rect[2]
    height = rect[3]
    return image[y:y + height, x:x + width]


# rectで指定した矩形の左上の座標を返す
def top_left(rect):
    return (rect[0], rect[1])


# rectで指定した矩形の右下の座標を返す
def bottom_right(rect):
    return (rect[0] + rect[2], rect[1] + rect[3])

# imageで指定した画像にrectの矩形を描画して返す
def drawRect(image, rect):
    return cv2.rectangle(image,
                         top_left(rect),
                         bottom_right(rect),
                         (0, 255, 0),
                         3)

def main():
    keyword = sys.argv[1]
    max_num = sys.argv[2]

    input_file_path = f"./training_data/original/{keyword}/"
    output_file_path = f"./training_data/cropped_face/{keyword}/"

    crawler = GoogleImageCrawler(storage={"root_dir": input_file_path})
    crawler.crawl(keyword=keyword, max_num=int(max_num))

    os.makedirs(output_file_path, exist_ok=True)
    input_files = os.listdir(input_file_path)

    # 顔認識用特徴量ファイルを読み込む --- （カスケードファイルのパスを指定）
    cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

    for input_file in input_files:
        input_image = cv2.imread(input_file_path + input_file)

        # 認識結果表示用のwindowを作成
        cv2.namedWindow('window', cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
        cv2.imshow('window', input_image)
        cv2.waitKey(100)

        # 顔認識の実行
        face_rects = cascade.detectMultiScale(input_image,
                                              scaleFactor=1.1,
                                              minNeighbors=10,
                                              minSize=(10, 10))

        # 顔の検出に失敗した場合はcontinue
        if len(face_rects) == 0:
            continue

        # 最初に検出した顔のみ取得
        face_rect = face_rects[0]

        # 顔領域だけ切り出して保存
        output_image = crop(input_image, face_rect)
        cv2.imwrite(output_file_path + input_file, output_image)

        # 顔領域に矩形を描画して表示
        input_image_with_rect = drawRect(input_image, face_rect)
        cv2.imshow('window', input_image_with_rect)
        cv2.waitKey(100)

        # windowを破棄
        cv2.destroyAllWindows()


main()
