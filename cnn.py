import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torch
from matplotlib import cm
import torchvision
from torchsummary import summary
import torchvision.transforms as transforms
from skimage import io, transform
import numpy as np
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Dropout2d, Flatten
from torch.optim import Adam, SGD
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import os
import time
import seaborn as sn



def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    #TP = conf_matrix.diag()
    #for c in range(15):
        #idx = torch.ones(15).byte()
        #idx[c] = 0
        #TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()
        #FP = conf_matrix[c, idx].sum()
        #FN = conf_matrix[idx, c].sum()

        #sensitivity = (TP[c] / (TP[c]+FN))
        #specificity = (TN / (TN+FP))

        #print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
        #    c, TP[c], TN, FP, FN))
        #print('Sensitivity = {}'.format(sensitivity))
        #print('Specificity = {}'.format(specificity))
    return conf_matrix

if not os.path.exists("output"):
    print("Data folder does not exist in this directory.")
    exit()
if not os.path.exists("output/train"):
    print("The training images are not existent inside the data folder.")
    exit()
if not os.path.exists("output/test"):
    print("The training images are not existent inside the data folder.")
    exit()

print("Found all data folders.", flush=True)
custom_transform = transforms.Compose([transforms.Resize([1020, 300]),
                                       # transforms.Grayscale(),
                                       transforms.ToTensor()])
train_data = ImageFolder(root='./output/train', transform=custom_transform)
test_data = ImageFolder(root='./output/test', transform=custom_transform)

loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)
print("Data loaded into data loaders.", flush=True)
# model = torchvision.models.vgg16(pretrained=True)
# # model = torchvision.models.densenet161(pretrained=True,  progress=True)
# for param in model.parameters():
#     param.requires_grad = False
# model.classifier[6] = Sequential(
#     Linear(4096, 1024),
#     ReLU(),
#     Linear(1024, 512),
#     ReLU(),
#     Linear(512, 11),
#     Softmax(dim=1)
# )
# for param in model.classifier[6].parameters():
#     param.requires_grad = True
model = torchvision.models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = Linear(num_ftrs, 11)

# load the model into memory
# model.load_state_dict(torch.load("./weights/0.hdf5"))
loaded_model = False
print("Model compiled.", flush=True)



optimizer = Adam(model.parameters(), lr=0.0001)
criterion = CrossEntropyLoss()
if torch.cuda.is_available():
    model.cuda()
    criterion = criterion.cuda()
summary(model, (3, 1020, 300))
epoch_count = 200
if not loaded_model:
    # tensorboard
    writer = SummaryWriter()
    print("Starting to train.", flush=True)
    best_val_acc = 0
    for epoch in range(epoch_count):
        start_time = time.time()
        # print(epoch)
        # adjust_learning_rate(optimizer, epoch, 0.01)

        ##################################################################
        # TRAINING SECTION
        ##################################################################

        model.train()
        correct, total, loss_total = 0, 0, 0
        for i, data in enumerate(loader, 0):
            # print('train ',i)
            input, label = data
            input, label = input.cuda(), label.cuda()

            optimizer.zero_grad()
            output = model(input)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            correct += (predicted.long() == label).float().sum()
            total += label.size(0)
            loss_total += loss.item()

        ##################################################################
        # VALIDATION SECTION
        ##################################################################

        model.eval()
        val_correct, val_total, val_loss_total = 0, 0, 0

        conf_matrix = torch.zeros(11, 11)
        predictions = []
        labels = []
        for i, data in enumerate(test_loader, 0):
            input, label = data
            input, label = input.cuda(), label.cuda()

            output = model(input)
            loss = criterion(output, label)

            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.tolist())
            labels.extend(label.tolist())

            val_correct += (predicted.long() == label).float().sum()
            val_total += label.size(0)
            val_loss_total += loss.item()

            conf_matrix = confusion_matrix(output, label, conf_matrix)
        # print(predictions)
        # print(label)
        # print statistics and store for tensorboard
        print("Epoch {}/{}".format(epoch, epoch_count))
        print("{}s - acc: {:.2f}% - loss: {:.5f} - val_acc: {:.2f}% - loss: {:.5f}".format(
            int(time.time()-start_time),
            100*correct/total,
            loss_total/total,
            100*val_correct/val_total,
            val_loss_total/val_total), flush=True)

        # torch.save(model.state_dict(), "./weights/{}.hdf5".format(epoch))
        """
        if (100*val_correct/val_total) > best_val_acc:
            best_val_acc = (100*val_correct/val_total)
            all = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', 'Industrial',
                   'Stadium', 'Underwater', 'TallBuilding', 'Street', 'Highway', 'Field', 'Coast',
                   'Mountain', 'Forest']
            categories = [x.lower() for x in all]
            cats = ["bedroom", "coast", "field", "forest", "highway", "house", "industrial", "kitchen",
                    "livingroom", "mountain", "stadium", "store", "street", "tallbuilding", "underwater"]
            actual_predictions = [[] for elem in all]

            
            for i in range(len(predictions)):
                prediction = predictions[i]
                label = labels[i]

                # "bedroom"
                target = cats[label]
                guessed = cats[prediction]

                actual_predictions[categories.index(target)].append(
                    all[categories.index(guessed)])
            
            actual_predictions = [
                pred for category in actual_predictions for pred in category]
            
            scipy.io.savemat('./cnn.mat', mdict={
                'predicted_categories': actual_predictions
            })
            print("Saved results to mat file")
            """
    print(conf_matrix)
    index=["FA1", "FA2", "FA3", "FS1", "FS2", "J", "MA1", "MA2", "MS1", "MS2", "MS3"]
    # plt.figure(figsize=(10,7))
    # df_cm = pd.DataFrame(conf_matrix, index=["FA1", "FA2", "FA3", "FS1", "FS2", "J", "MA1", "MA2", "MS1", "MS2", "MS3"], columns=["FA1", "FA2", "FA3", "FS1", "FS2", "J", "MA1", "MA2", "MS1", "MS2", "MS3"])
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(conf_matrix, annot=True, annot_kws={"size": 16}, cmap='Greens', xticklabels=index, yticklabels=index, fmt='g') # font size
    plt.show()
    print("Training finished.")
