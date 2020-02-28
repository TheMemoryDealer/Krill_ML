# Simple CNN for the MNIST Dataset
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.applications.resnet50 import preprocess_input
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import os
from PIL import Image
from keras.optimizers import Adam
import numpy as np
import csv


store = {}

i = 0
for filename in os.listdir('Krill_images'):
    image = Image.open('./Krill_images/{}'.format(filename))  # .convert('L')

    new_image = image.resize((300, 100))

    img = np.array(new_image)
    newname = str(filename).replace(".jpg", "")

    img = preprocess_input(img)

    store[newname] = {}
    store[newname]["img"] = img
    if i % 500 == 0 and i is not 0:
        print("Completed reading in image", i)
    i += 1

with open("krill_data_set.csv") as csv_file:
    csv = csv.reader(csv_file, delimiter=',')
    i = 0
    for row in csv:
        if i == 0:
            i += 1
            continue
        try:
            if str(row[1]) == '' or str(row[1]) == 'U':
                continue
            store[row[8]]["target"] = str(row[1])
        except:
            continue

X_train = []
labels = []

for key, value in store.items():
    if "target" not in store[key]:
        continue
    X_train.append(store[key]["img"])
    labels.append(store[key]["target"])

num_fucks = list(set(labels))
print(np.shape(X_train))
print(np.shape(labels))
print(num_fucks)

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

#X_test = X_train[int(len(X_train)-len(X_train)/20):len(X_train)]
#X_train = X_train[0:int(len(X_train)-len(X_train)/20)]
#y_test = y_train[int(len(y_train)-len(y_train)/20):len(y_train)]
#y_train = y_train[0:int(len(y_train)-len(y_train)/20)]

X_train = X_train.reshape((X_train.shape[0], 300, 100, 3)).astype('float32')
#X_test = X_test.reshape((X_test.shape[0], 64, 64, 1)).astype('float32')
# # print(X_train)
#X_train = X_train / 255
#X_test = X_test / 255


def baseline_model(lr):
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
    return model


def resnet(lr):
    from keras.applications.resnet50 import ResNet50
    from keras.layers import Input

    model = Sequential()
    image_input = Input(shape=(90, 30, 3))
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
    model = densenet()
    # model = resnet()
    #model = densenet()

    history = model.fit(X_train, y_train, validation_split=0.2,
                        epochs=300, batch_size=32, shuffle=True)

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


mnist_fork_single()
