import os
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import glob


#ディレクトリを作成
if not os.path.exists("result"):
    os.mkdir("result")
dirname = "./result/"
#modelの読み込み
model = load_model("./arashi_model14.h5")
#適用する画像があるディレクトリを開く
img_path_list = glob.glob("test/*")
num = 0
for img_path in img_path_list:
        img = cv2.imread(img_path, 1)
        name,ext = os.path.splitext(img_path)
        num += 1
        file_name = dirname + "pic" +  str(num) + str(ext)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = "./haarcascade_frontalface_alt.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        #顔認識を実行
        faces=cascade.detectMultiScale(img_gray)
        Labels=["MJ","aiba","ono","ninomiya","sakurai"]

        Threshold = 0.01
        #顔が検出されたとき
        if len(faces) > 0:
            print(len(faces))
            for fp in faces:
                # 学習したモデルでスコアを計算する
                img_face = img[fp[1]:fp[1]+fp[3], fp[0]:fp[0]+fp[2]]
                img_face = cv2.resize(img_face, (50, 50))
                score = model.predict(np.expand_dims(img_face, axis=0))
                # 最も高いスコアを書き込む
                score_argmax = np.argmax(np.array(score[0]))
                print(score_argmax )
                #閾値以下で表示させない
                if score[0][score_argmax] < Threshold:
                    continue
                #文字サイズの調整
                fs_rate= 0.008
                text =  "{0} {1:.1f}% ".format(Labels[score_argmax], score[0][score_argmax]*100)
                #文字を書く座標の調整
                text_rate = 0.22
                #ラベルを色で分ける
                #cv2なのでBGR
                if Labels[score_argmax] == "MJ":
                    cv2.rectangle(img, tuple(fp[0:2]), tuple(fp[0:2]+fp[2:4]), (255, 0, 255), thickness=3)
                    cv2.putText(img, text, (fp[0],fp[1]+fp[3]+int(fp[3]*text_rate)),cv2.FONT_HERSHEY_DUPLEX,(fp[3])*fs_rate, (255,0,255), 2)
                if Labels[score_argmax] == "aiba":
                    cv2.rectangle(img, tuple(fp[0:2]), tuple(fp[0:2]+fp[2:4]), (0, 255, 0), thickness=3)
                    cv2.putText(img, text, (fp[0],fp[1]+fp[3]+int(fp[3]*text_rate)),cv2.FONT_HERSHEY_DUPLEX,(fp[3])*fs_rate, (0,255,0), 2)
                if Labels[score_argmax] == "ono":
                    cv2.rectangle(img, tuple(fp[0:2]), tuple(fp[0:2]+fp[2:4]), (255, 0, 0), thickness=3)
                    cv2.putText(img, text, (fp[0],fp[1]+fp[3]+int(fp[3]*text_rate)),cv2.FONT_HERSHEY_DUPLEX,(fp[3])*fs_rate, (255,0,0), 2)
                if Labels[score_argmax] == "ninomiya":
                    cv2.rectangle(img, tuple(fp[0:2]), tuple(fp[0:2]+fp[2:4]), (0, 255, 255), thickness=3)
                    cv2.putText(img, text, (fp[0],fp[1]+fp[3]+int(fp[3]*text_rate)),cv2.FONT_HERSHEY_DUPLEX,(fp[3])*fs_rate, (0,255,255), 2)
                if Labels[score_argmax] == "sakurai":
                    cv2.rectangle(img, tuple(fp[0:2]), tuple(fp[0:2]+fp[2:4]), (0, 0, 255), thickness=3)
                    cv2.putText(img, text, (fp[0],fp[1]+fp[3]+int(fp[3]*text_rate)),cv2.FONT_HERSHEY_DUPLEX,(fp[3])*fs_rate, (0,0,255), 2)
                plt.figure(figsize=(4, 4),dpi=200)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()
            cv2.imwrite(file_name, img)
        # 顔が検出されなかったとき
        else:
            print("no face")