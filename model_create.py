import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Activation, Dense, Dropout,Flatten, Input
from keras.models import Sequential, load_model,Model
from keras import optimizers
from keras.utils import to_categorical


DATADIR = "./act"
CATEGORIES = ["matu", "ai", "ono" , "nino" , "saku"]
IMG_SIZE = 50
training_data = []
def create_training_data():
    for class_num, category in enumerate(CATEGORIES):
        path = os.path.join(DATADIR, category)
        for image_name in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image_name),)  # 画像読み込み
                img_resize_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # 画像のリサイズ（必要な場合）
                training_data.append([img_resize_array, class_num])  # 画像データ、ラベル情報を追加
            except Exception as e:
                pass

create_training_data()#関数create_training_dataでラベル付する

random.shuffle(training_data)  # データをシャッフル
X_trains = []  # 画像データ
y_trains = []  # ラベル情報
# データセット作成
for feature, label in training_data:
    X_trains.append(feature)
    y_trains.append(label)
# numpy配列に変換
X = np.array(X_trains)
y = np.array(y_trains)




'''
# データセットの確認
for i in range(0, 12):
    print("学習データのラベル：", y_trains[i])
    plt.subplot(4, 4, i+1)
    plt.axis('off')
    if y_trains[i] ==0:
        plt.title(label="MJ")
    elif y_trains[i] ==1:
        plt.title(label = "aiba")
    elif y_trains[i] ==2:
        plt.title(label = "ono")
    elif y_trains[i] ==3:
        plt.title(label = "nino")
    elif y_trains[i] ==4:
        plt.title(label = "sakurai")
    #plt.title(label = 'MJ' if y_trains[i] == 0 else 'nagato')
    img_array = cv2.cvtColor(X_trains[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img_array)
plt.show()
'''

print(len(X))
print(len(y))


#データの分割
X_train = X[:5000]
X_test = X[5000:]
train_y = y[:5000][:]
test_y = y[5000:][:]


y_train = to_categorical(train_y)
y_test = to_categorical(test_y)

# vgg16のインスタンスの生成
input_tensor = Input(shape=(50, 50, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(64,activation="relu"))
top_model.add(Dropout(0.1))
top_model.add(Dense(32,activation="relu"))
top_model.add(Dense(5,  activation='softmax'))

# vgg16とtop_modelを連結
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
# 19層目までの重みをfor文を用いて固定
for layer in model.layers[:19]:
  layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=64, epochs=100,validation_data=(X_test, y_test))


# 精度の評価
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#modelの保存
model.save("arashi_model14.h5")
#epoch毎の予測値の正解データとの誤差を表している
#バリデーションデータのみ誤差が大きい場合、過学習を起こしている

loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=len(loss)

plt.plot(range(epochs), loss, marker = '.', label = 'loss')
plt.plot(range(epochs), val_loss, marker = '.', label = 'val_loss')
plt.legend(loc = 'best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



