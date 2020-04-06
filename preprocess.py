import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import ResNet50
from keras.layers import Input
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.applications.resnet50 import preprocess_input
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import keras
import os
from PIL import Image
import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
import csv
import seaborn as sns
import pandas as pd
from tensorflow.python.client import device_lib
import tensorflow as tf
import shutil


store = {}

with open('./toMirror.txt') as f:
    lines = [line.rstrip('\n') for line in f]
    # print(lines[1])
    # exit()
#     for l in lines:
#         for filename in os.listdir('../Krill_images'):
#             if any(l in filename)

# exit()


i = 0
"""
Uncomment this when dataset is in one large folder rather than sub minis
"""
folders = []
for (dirpath, dirnames, filenames) in os.walk('./Krill_images'):
    folders.extend(dirnames)
    break
# print(f)
# exit()
for f in folders:
    # print(f)
    for filename in os.listdir('./Krill_images/{}'.format(f)):
        try:
            image = Image.open('./Krill_images/{}/{}'.format(f,filename))  # .convert('L')
            # print(filename)
            # print(lines[1])
            # print("!!!!!SS")
            for l in lines:
                if(l in filename):
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    # image.show()
            # if filename in lines:
            #     print("got one")
            #     image = image.transpose(Image.FLIP_LEFT_RIGHT)


            new_image = image.resize((300, 100))
            # new_image.show()
            # quit()
            img = np.array(new_image)
            newname = str(filename).replace(".jpg", "")
            # print(newname)
            img = preprocess_input(img)

            store[newname] = {}
            store[newname]["img"] = img
            store[newname]["filename"] = filename


            """
            Uncomment this if you want to cut the dataset to 1000
            """
            # if i % 1000 == 0 and i is not 0:
            #    break


            if i % 500 == 0 and i is not 0:
                print("Completed reading in image", i)
            i += 1
        except Exception as e:
            print(e)
            continue
# exit()
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
            # print(store[row[8]]["target"])
        except Exception as e:
            # print(str(row[8]))
            # print(e)
            continue

X_train = []
labels = []

Jcounter = 0
for key, value in store.items():
    if "target" not in store[key]:
        continue
    if store[key]["target"] == "J":
        Jcounter += 1
    # print(store[key]["target"])
    # print(Jcounter)
    if Jcounter > 1065 and store[key]["target"] == "J":
        continue
    else:
        X_train.append(store[key]["img"])
        labels.append(store[key]["target"])

# for lab in labels:
#     print(lab)

"""
This moves all wanted targets to own folders
"""
#     if os.path.exists("../Krill_images/{}".format(store[key]["target"])):
#         shutil.move("../Krill_images/{}".format(store[key]["filename"]), "../Krill_images/{}/{}".format(store[key]["target"], store[key]["filename"]))

num_fucks = list(set(labels))
print(np.shape(X_train))
print(np.shape(labels))
print(num_fucks)
print(tf.__version__)
print(keras.__version__)
# print(device_lib.list_local_devices())
# print(tf.test.is_built_with_cuda())
# print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
# print(device_lib.list_local_devices())
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

density = np.zeros(len(num_fucks))
for index in y_train:
    density[np.argmax(index)] += 1

# print(len(density), len(num_fucks))
plt.bar(num_fucks, density)
plt.show()
exit()
X_test = X_train[int(len(X_train)-len(X_train)/20):len(X_train)]
X_train = X_train[0:int(len(X_train)-len(X_train)/20)]
y_test = y_train[int(len(y_train)-len(y_train)/20):len(y_train)]
y_train = y_train[0:int(len(y_train)-len(y_train)/20)]

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
    model = Sequential()
    image_input = Input(shape=(90, 30, 3))
    model.add(ResNet50(input_tensor=image_input, weights='imagenet'))
    model.add(Dense(len(num_fucks), activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=["accuracy"],
                  optimizer=Adam(lr=lr, decay=1e-6))
    return model


def densenet():
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

    learning_rates = [0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    accuracies = []
    val_accs = []
    losses = []
    val_losses = []
    for lr in learning_rates:
        model = densenet(lr)

        history = model.fit(X_train, y_train, validation_split=0.2,
                            epochs=500, batch_size=32, shuffle=True)
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
    history = model.fit(X_train, y_train,  validation_split=0.2, epochs=100, batch_size=32, shuffle=True)

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


mnist_fork_single()
