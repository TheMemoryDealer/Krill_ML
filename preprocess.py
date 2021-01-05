# Simple CNN for the MNIST Dataset
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.applications.resnet50 import preprocess_input
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import os
from PIL import Image
import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
import csv
import seaborn as sns
import pandas as pd
import cv2
import shutil
import split_folders
import random
store = {}


# # Split with a ratio.
# # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
# split_folders.ratio('../../Krill_images/Krill images all/+paded+reflected+targets_full', output="output", seed=1337, ratio=(.8, .2)) # default values




# # exit()
# with open('./toMirror.txt') as f:
#     lines = [line.rstrip('\n') for line in f]
#     # print(lines[1])
#     # exit()
# #     for l in lines:
# #         for filename in os.listdir('../Krill_images'):
# #             if any(l in filename)

# # exit()
"""
Uncomment this when dataset is in one large folder rather than sub minis
"""

# print(f)
# exit()
# .listdir('./Krill_images/{}'.format(f)):





folders = []
for (dirpath, dirnames, filenames) in os.walk('./outputLATtest/train'):
    folders.extend(dirnames)
    break
for f in folders:
    i = 0
    for filename in os.listdir('./outputLATtest/train/{}'.format(f)):
        if "Copy" in filename:
            img = cv2.imread('./outputLATtest/train/{}/{}'.format(f,filename))  # .convert('L')
            # cv2.imshow("img",img) 

            #-----Converting image to LAB Color model----------------------------------- 
            lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            # cv2.imshow("lab",lab)

            #-----Splitting the LAB image to different channels-------------------------
            l, a, b = cv2.split(lab)
            # cv2.imshow('l_channel', l)
            # cv2.imshow('a_channel', a)
            # cv2.imshow('b_channel', b)

            #-----Applying CLAHE to L-channel-------------------------------------------
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            # cv2.imshow('CLAHE output', cl)

            #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
            limg = cv2.merge((cl,a,b))
            # cv2.imshow('limg', limg)

            #-----Converting image from LAB Color model to RGB model--------------------
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            img = final
            # cv2.imshow('final', final)
            # img = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)  # cv2 defaul color code is BGR
            h,w,c = img.shape # (768, 1024, 3)

            noise = np.random.randint(0,50,(h, w)) # design jitter/noise here
            zitter = np.zeros_like(img)
            zitter[:,:,1] = noise  

            noise_added = cv2.add(img, zitter)
            c = int(h/2)
            combined = np.vstack((img[:c,:,:], noise_added[c:,:,:]))
            hsvImg = cv2.cvtColor(noise_added,cv2.COLOR_RGB2HSV)
            r = random.uniform(0.3, 1)
            # print(r)
            # decreasing the V channel by a factor from the original
            hsvImg[...,2] = hsvImg[...,2]*r
            # plt.subplot(111), plt.imshow(cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB))
            # cv2.imshow("result", cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite('./outputLATtest/train/{}/{}'.format(f,filename), cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB))
        # exit()
        # ht, wd, cc= image.shape
        # print(filename)


        # if any(x in filename for x in lines):
        #     # print(filename)
        #     image = cv2.flip(image, 1)
        #     cv2.imwrite('../../Krill_images/Krill images all/+paded+reflected/{}'.format(filename), image)
    
        # for l in lines:
        #     if(l in filename):
        #         print("got image")
        #         image = cv2.flip(image, 1)
                # cv2.imshow('image',image)
                # cv2.waitKey(0)
                # exit()

        # exit()
        # # create new image of desired size and color (blue) for padding
        # ht, wd, cc= image.shape
        # ww = 1700
        # hh = 500
        # color = (245,127,56)
        # result = np.full((hh,ww,cc), color, dtype=np.uint8)

        # # compute center offset
        # xx = (ww - wd) // 2
        # yy = (hh - ht) // 2

        # # copy img image into center of result image
        # result[yy:yy+ht, xx:xx+wd] = image
        # # print(filename)
        # cv2.imwrite('../../Krill images lateral/paded/{}'.format(filename), result)
        # # print("done")
        # view result
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # result.show()
        # exit()
        # save result
        # new_image = cv2.resize(image, (340, 100))
        # # cv2.imshow("new_image",new_image)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        # # new_image.show()
        # # exit()
        # img = np.array(new_image)
        # newname = str(filename).replace(".jpg", "")
        # # print(newname)
        # # img = preprocess_input(img)

        # store[newname] = {}
        # store[newname]["img"] = img
        # store[newname]["filename"] = filename

        # store[newname]["alt"] = filename
        # #if i % 3000 == 0 and i is not 0:
        #    break
        if i % 500 == 0 and i is not 0:
            print("Completed reading in image", i)
        i += 1
print("done reading iamges_____")
exit()
with open("krill_data_set.csv") as csv_file:
    dontwant = ['M1',"FA5","A2","FS3","MA3","","U"]
    csv = csv.reader(csv_file, delimiter=',')
    i = 0
    for row in csv:
        # print(Jcounter)
        if i == 0:
            i += 1 # skip csv header
            continue
        try:
            if str(row[1]) in dontwant:
                # print(str(row[8]))
                continue
            store[row[8]]["target"] = str(row[1])
            store[row[8]]["alt"] = str(row[9]) + '.jpg'
            # print(store[row[8]]["target"])
        except Exception as e:
            # print(str(row[8]))
            # print(e)
            continue

X_train = []
labels = []
# print(store)
# yz = 0
# for key, value in store.items():
#     try:
#         altimg = cv2.imread('../../Krill_images/Krill images all concat/+paded+reflected/{}'.format(store[key]["alt"]))
#         img = cv2.imread('../../Krill_images/Krill images all concat/+paded+reflected/{}'.format(store[key]["filename"]))
#         os.remove('../../Krill_images/Krill images all concat/+paded+reflected/{}'.format(store[key]["alt"]))
#         os.remove('../../Krill_images/Krill images all concat/+paded+reflected/{}'.format(store[key]["filename"]))
#     except Exception as e:
#         continue
#     yz += 1
#     print(yz)
#     h_img = cv2.vconcat([img, altimg])
#     cv2.imwrite('../../Krill_images/Krill images all concat/+paded+reflected+concat/{}'.format(store[key]["filename"]), h_img)


# exit()
Jcounter = 0
for key, value in store.items():
    if "target" not in store[key]:
        continue
    if store[key]["target"] == "J":
        Jcounter += 1
    # print(store[key]["target"])
    # print(Jcounter)
    if Jcounter > 1000 and store[key]["target"] == "J":
        continue
    else:
        X_train.append(store[key]["img"])
        labels.append(store[key]["target"])


    if os.path.exists("../../Krill_images/Krill images all/+paded+reflected+targets/{}".format(store[key]["target"])):
        shutil.move("../../Krill_images/Krill images all/+paded+reflected/{}".format(store[key]["filename"]), 
        "../../Krill_images/Krill images all/+paded+reflected+targets/{}/{}".format(store[key]["target"], store[key]["filename"]))


num_fucks = list(set(labels))
print(np.shape(X_train))
print(np.shape(labels))
print(num_fucks)


# exit()
y_train = []
for y in labels:
    category = num_fucks.index(y)
    replaced = np.zeros(len(num_fucks))
    replaced[category] = 1
    y_train.append(replaced)

# X_train.append(img)

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.array(X_train)
y_train = np.array(y_train)
plt.rcParams.update({'font.size': 22})

density = np.zeros(len(num_fucks))
for index in y_train:
    density[np.argmax(index)] += 1

print(len(density), len(num_fucks))
plt.bar(num_fucks, density) 
plt.show()
exit()
# X_test = X_train[int(len(X_train)-len(X_train)/20):len(X_train)]
# X_train = X_train[0:int(len(X_train)-len(X_train)/20)]
# y_test = y_train[int(len(y_train)-len(y_train)/20):len(y_train)]
# y_train = y_train[0:int(len(y_train)-len(y_train)/20)]

#X_train = X_train.reshape((X_train.shape[0], 300, 100, 3)).astype('float32')
#X_test = X_test.reshape((X_test.shape[0], 64, 64, 1)).astype('float32')

def baseline_model(lr):
    """
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(90, 30, 3), activation='relu'))
    # model.add(Conv2D(64, (5, 5), input_shape=(64, 64, 3), activation='relu'))
    # model.add(Conv2D(128, (5, 5), input_shape=(64, 64, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(len(num_fucks), activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=["accuracy"],
                  optimizer=Adam(lr=lr, decay=1e-6))
    """

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(num_fucks), activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=["accuracy"],
                  optimizer=Adam(lr=lr, decay=1e-6))
    return model


def resnet(lr):
    from keras.applications.resnet50 import ResNet50
    from keras.layers import Input

    model = Sequential()
    image_input = Input(shape=(100, 340, 3))
    model.add(ResNet50(input_tensor=image_input, weights='imagenet'))
    model.add(Dense(len(num_fucks), activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=["accuracy"],
                  optimizer=Adam(lr=lr, decay=1e-6))
    return model


def densenet():
    from keras.applications.densenet import DenseNet121
    from keras.layers import Input

    model = Sequential()
    image_input = Input(shape=(300, 100, 3))
    model.add(DenseNet121(input_tensor=image_input, weights='imagenet'))
    model.add(Dense(len(num_fucks), activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=["accuracy"],
                  optimizer=Adam(lr=0.00001, decay=1e-6))
    return model


def mnist_fork():
    #model = baseline_model()
    # model = resnet()
    #model = densenet()

    learning_rates = [0.0001]
    accuracies = []
    val_accs = []
    losses = []
    val_losses = []
    for lr in learning_rates:
        model = resnet(lr)

        history = model.fit(X_train, y_train, validation_split=0.2,
                            epochs=100, batch_size=36, shuffle=True)
        accuracies.append(history.history['accuracy'])
        val_accs.append(history.history['val_accuracy'])
        losses.append(history.history['loss'])
        val_losses.append(history.history['val_loss'])

    fig, ax = plt.subplots(ncols=2, nrows=2)

    for i in range(len(learning_rates)):
        lr = learning_rates[i]

        ax[0, 0].plot(accuracies[i], label="training [{}]".format(lr))
        ax[0, 1].plot(val_accs[i], label="validation [{}]".format(lr))

        ax[1, 0].plot(losses[i], label="training [{}]".format(lr))
        ax[1, 1].plot(val_losses[i], label="validation [{}]".format(lr))

    ax[0, 0].set_title('model accuracy')
    ax[0, 0].set_ylabel('accuracy')
    ax[0, 0].set_xlabel('epoch')
    ax[0, 0].legend(loc='upper left')
    ax[0, 1].set_title('model accuracy')
    ax[0, 1].set_ylabel('accuracy')
    ax[0, 1].set_xlabel('epoch')
    ax[0, 1].legend(loc='upper left')

    ax[1, 0].set_title('model losses')
    ax[1, 0].set_ylabel('loss')
    ax[1, 0].set_xlabel('epoch')
    ax[1, 0].legend(loc='upper left')
    ax[1, 1].set_title('model losses')
    ax[1, 1].set_ylabel('loss')
    ax[1, 1].set_xlabel('epoch')
    ax[1, 1].legend(loc='upper left')

    plt.show()


def mnist_fork_single():
    model = baseline_model(0.0001)
    # model = resnet()
    #model = densenet()

    # history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, shuffle=True)
    history = model.fit(X_train, y_train,  validation_split=0.2, epochs=30, batch_size=8, shuffle=True)

    predictions = model.predict_classes(X_test)
    y_targets = [np.argmax(item) for item in y_test]

    cf = multilabel_confusion_matrix(y_targets, predictions)

    con_mat = tf.math.confusion_matrix(labels=y_targets, predictions=predictions).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm,
                        index=num_fucks, 
                        columns=num_fucks)
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    """
    fig, ax = plt.subplots(ncols=2, nrows=1)
    ax[0].plot(history.history['accuracy'], label="training")
    ax[0].plot(history.history['val_accuracy'], label="validation")
    ax[0].set_title('model accuracy')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].legend(loc='upper left')
    ax[1].plot(history.history['loss'], label="training")
    ax[1].plot(history.history['val_loss'], label="validation")
    ax[1].set_title('model accuracy')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(loc='upper left')
    plt.show()
    """


mnist_fork()