import os
import glob
import numpy as np
import cv2

def scratch_image(img, flip=True, thr=True, filt=False, resize=False, erode=False):
    # 水増しの手法を配列にまとめる
    methods = [flip, thr, filt, resize, erode]

    # flip は画像の左右反転
    # thr  は閾値処理
    # filt はぼかし
    # resizeはモザイク
    # erode は収縮
    #     をするorしないを指定している
    # 
    # imgの型はOpenCVのcv2.read()によって読み込まれた画像データの型
    # 
    # 水増しした画像データを配列にまとめて返す

    # 画像のサイズを習得、収縮処理に使うフィルターの作成
    img_size = img.shape
    filter1 = np.ones((3, 3))
    # オリジナルの画像データを配列に格納
    images = [img]

    # 手法に用いる関数
    scratch = np.array([

        #画像の左右反転
        lambda x: cv2.flip(x, 1),

        #閾値処理
        lambda x: cv2.threshold(x,150,255,cv2.THRESH_TOZERO)[1],

        #ぼかし
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),

        #モザイク処理
        lambda x:cv2.resize(cv2.resize(x,(img_size[1]//5,img_size[0]//5)),(img_size[1],img_size[0])),

        #収縮
        lambda x:cv2.erode(x,filter1)

    ])

    # 関数と画像を引数に、加工した画像を元と合わせて水増しする関数
    doubling_images = lambda f, imag: (imag + [f(i) for i in imag])

    # doubling_imagesを用いてmethodsがTrueの関数で水増し
    for func in scratch[methods]:
        images = doubling_images(func,images)

    return images
'''
#1200枚の画像をfor文で回すため、名称リストを作る。
lists = list(range(1,1201))
list_name = []
for a in lists:
    list_name.append("x" +str(a))

for a in listb:   

    # 画像の読み込み
    cat_img = cv2.imread("./loop8/"+a+".png",1)

    # 画像の水増し
    scratch_cat_images = scratch_image(cat_img)

    # 画像を保存するフォルダーを作成
    if not os.path.exists("scratch_images"):
        os.mkdir("scratch_images")

    for num, im in enumerate(scratch_cat_images):
        # まず保存先のディレクトリ"scratch_images/"を指定、番号を付けて保存
        cv2.imwrite("scratch_images/" + str(a)+str(num) + ".png" ,im) 
'''


for a in range(1,46):
    # 画像の読み込み
    cat_img = cv2.imread("./act/相葉/"+str(a)+"x.png",1)
    # 画像の水増し
    scratch_cat_images = scratch_image(cat_img)

    for num, im in enumerate(scratch_cat_images):
      # まず保存先のディレクトリ"scratch_images/"を指定、番号を付けて保存
        cv2.imwrite("./act/相葉２/" +str(a)+str(num) + ".png" ,im) 
        print(im)
