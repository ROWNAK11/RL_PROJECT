#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

from keras import callbacks, optimizers
from keras.models import Model
from keras.applications import ResNet50, InceptionV3, MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint
#import torch
#import ImageHelper

#from ImageHelper import NumpyImg2Tensor


# In[2]:


class ConvolutionalNeuralNetworks:
    def __init__(self, networkName, datasetInfo):
        self.datasetInfo = datasetInfo
        self.networkName = networkName
        self.model = None
        self.last_base_layer_idx = 0
        self.callbacks = [
            callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
            ModelCheckpoint(os.path.join('models', 'best' + self.networkName + self.datasetInfo + '.hdf5'),
                            monitor='val_accuracy')]

    def __get_layer_idx_by_name(self, layerName):
        index = None
        for idx, layer in enumerate(self.model.layers):
            if layer.name == layerName:
                index = idx
                break
        return index

    def create_model_architecture(self, shape=(64, 64, 3)):
        if self.networkName == "ResNet":
            self.model = ResNet50(include_top=False, weights="imagenet", input_shape=shape)
            self.last_base_layer_idx = self.__get_layer_idx_by_name(self.model.layers[-1].name)
            for layer in self.model.layers[:-18]:
                layer.trainable = False
            gavp = GlobalAveragePooling2D()(self.model.output)
            d1 = Dense(1024, 'relu')(gavp)
            d2 = Dense(1024, 'relu')(d1)
            d3 = Dense(1024, 'relu')(d2)
            d4 = Dense(512, 'relu')(d3)
            d5 = Dense(7, 'softmax')(d4)
            self.model = Model(inputs=self.model.input, outputs=d5)
        if self.networkName == "Inception":
            self.model = InceptionV3(include_top=False, weights="imagenet", input_shape=shape)
            self.last_base_layer_idx = self.__get_layer_idx_by_name(self.model.layers[-1].name)
            for layer in self.model.layers[:-4]:
                layer.trainable = False
            f = Flatten()(self.model.output)
            d1 = Dense(1024, 'relu')(f)
            do1 = Dropout(0.2)(d1)
            d2 = Dense(7, 'softmax')(do1)
            self.model = Model(inputs=self.model.input, outputs=d2)
        if self.networkName == 'MobileNet':
            self.model = MobileNetV2(include_top=False, input_shape=shape)
            self.last_base_layer_idx = self.__get_layer_idx_by_name(self.model.layers[-1].name)
            for layer in self.model.layers[:-4]:
                layer.trainable = False
            gavp = GlobalAveragePooling2D()(self.model.output)
            dense = Dense(7, 'softmax')(gavp)
            self.model = Model(inputs=self.model.input, outputs=dense)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

    def get_output_base_model(self, img):
        feature_extractor = Model(inputs=self.model.inputs,
                                  outputs=[layer.output for layer in self.model.layers])
        features = feature_extractor(NumpyImg2Tensor(img))
        return features[self.last_base_layer_idx]


# In[3]:


import numpy as np
from PIL import Image


def ShowNumpyImg(numpyImg):
    Image.fromarray(numpyImg, 'RGB').show()


def NumpyImg2Tensor(numpyImg):
    return np.expand_dims(numpyImg, axis=0)


# In[8]:


import os
from keras.utils.np_utils import to_categorical
import simplejson
import numpy as np
from PIL import Image
import glob


class DataLoader:
    def __init__(self, path, extension, classes, img_size, limit):
        self.path = path
        self.extension = str(extension)
        self.classes = classes
        self.img_size = img_size
        self.limit = limit
        self.datasetInfo = '_' + str(self.img_size) + '_limit_' + str(self.limit)
        self.splitDatasetsDir = 'splitDatasets' + str(img_size)
        self.modelsDir = 'models'
        self.resultsDir = 'out'

    def load(self):
        images = []
        labels = []
        for character_dir in glob.glob(os.path.join(self.path, "*", "")):
            for emotion_dir in glob.glob(os.path.join(character_dir, "*", "")):
                emotion_dir_name = os.path.basename(os.path.normpath(emotion_dir))
                emotion_name = emotion_dir_name.split("_")[1]
                emotion_idx = self.classes[emotion_name]  # make one-hot from this
                i = 0
                for img_name in glob.glob(os.path.join(emotion_dir, "*" + self.extension)):
                    if self.limit and i > self.limit:
                        break
                    img = Image.open(img_name).resize((self.img_size, self.img_size))
                    # removing the 4th dim which is transparency and rescaling to 0-1 range
                    im = np.array(img)[..., :3]
                    images.append(im)
                    labels.append(emotion_idx)
                    i += 1
        return np.array(images), np.array(labels)

    def demo_load(self):
        path1 = os.path.join(self.path, "C:\\Users\\tiwar\\Downloads\\FERG_DB_256\\malcolm\\malcolm_anger\\malcolm_anger_16.png")
        path2 = os.path.join(self.path, "C:\\Users\\tiwar\\Downloads\\FERG_DB_256\\malcolm\\malcolm_anger\\malcolm_anger_408.png")
        img1 = np.array(Image.open(path1).resize((self.img_size, self.img_size)))[..., :3]
        img2 = np.array(Image.open(path2).resize((self.img_size, self.img_size)))[..., :3]
        return np.array([img1, img2]), np.array([0, 0])

    def save_train_test_split(self, X_train, X_test, y_train, y_test):
        np.save(os.path.join(self.splitDatasetsDir, 'X_train_size' + self.datasetInfo + '.npy'), X_train)
        np.save(os.path.join(self.splitDatasetsDir, 'X_test_size' + self.datasetInfo + '.npy'), X_test)
        np.save(os.path.join(self.splitDatasetsDir, 'y_train_size' + self.datasetInfo + '.npy'), y_train)
        np.save(os.path.join(self.splitDatasetsDir, 'y_test_size' + self.datasetInfo + '.npy'), y_test)

    def load_train_test_split(self):
        X_train = np.load(os.path.join(self.splitDatasetsDir, 'X_train_size' + self.datasetInfo + '.npy'))
        X_test = np.load(os.path.join(self.splitDatasetsDir, 'X_test_size' + self.datasetInfo + '.npy'))
        y_train = np.load(os.path.join(self.splitDatasetsDir, 'y_train_size' + self.datasetInfo + '.npy'))
        y_test = np.load(os.path.join(self.splitDatasetsDir, 'y_test_size' + self.datasetInfo + '.npy'))
        return X_train, X_test, y_train, y_test

    def toOneHot(self, yData):
        return to_categorical(yData, num_classes=len(self.classes))

    def save_training_history(self, history):
        np.save(os.path.join(self.resultsDir, 'history' + self.datasetInfo + '.npy'), history)

    def load_training_history(self):
        return np.load(os.path.join(self.resultsDir, 'history' + self.datasetInfo + '.npy'), allow_pickle=True).item()

    def save_model(self, networkName, model):
        model_json = model.to_json()
        with open(os.path.join(self.modelsDir, 'model' + networkName + ".json"), "w") as json_file:
            json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))
        model.save_weights(os.path.join(self.modelsDir, 'model' + networkName + self.datasetInfo + '.h5'))
        print("Saved model to disk")

    def load_model_weights(self, networkName, model):
        model.load_weights(os.path.join(self.modelsDir, 'best' + networkName + self.datasetInfo + '.hdf5'))

    def save_details(self, stats, networkName, fileName="RL"):
        with open(os.path.join(self.resultsDir, 'details' + networkName + self.datasetInfo + fileName + ".txt"),
                  "w") as f:
            f.write("recall: " + str(stats.recall) + '\n')
            f.write("precision: " + str(stats.precision) + '\n')
            f.write("F1 score: " + str(stats.f1Score) + '\n')
            f.write("report: " + str(stats.report) + '\n')
            f.write("accuracy: " + str(stats.accuracy) + '\n')


# In[7]:


import os

from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd


def plot_history(dataLoader, networkName, history):
    plt.figure(figsize=(10, 7))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(dataLoader.resultsDir, 'train_history_loss' + networkName + dataLoader.datasetInfo + '.png'))
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(dataLoader.resultsDir, 'train_history_acc' + networkName + dataLoader.datasetInfo + '.png'))
    plt.show()


def plot_actions_stats(dataLoader, networkName, actions, stats, filename):
    plt.bar(actions, height=stats)
    plt.title('actions statistics')
    plt.ylabel('number of times action was chosen')
    plt.xlabel('action name')
    plt.savefig(os.path.join(dataLoader.resultsDir, 'actions_stats' + filename + networkName + dataLoader.datasetInfo +
                             '.png'))
    plt.show()


def plot_conf_matrix(dataLoader, networkName, conf_matrix, classes, filename):
    df_cm = pd.DataFrame(conf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(dataLoader.resultsDir, 'conf_matrix' + filename + networkName + dataLoader.datasetInfo +
                             '.png'))


def print_classification_details(statController):
    print("accuracy: ", statController.accuracy)
    print("precision: ", statController.precision)
    print("recall: ", statController.recall)
    print("F1 score: ", statController.f1Score)
    print("report: ", statController.report)


# In[13]:


#from ConvolutionalNeuralNetworks import ConvolutionalNeuralNetworks
#from DataLoader import DataLoader
#from Plotter import *
#from ImageHelper import NumpyImg2Tensor, ShowNumpyImg
#from QLearningModel import QLearningModel
from sklearn.model_selection import train_test_split
import time
#from StatisticsController import StatisticsController
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
import numpy as np

# set to true is algorithm is launched for the first time
LOAD_DATA = False
TRAIN_NETWORK = False
# limit of photos per 1 emotion per 1 person is 100
LIMIT = 10
ACTION_NAMES = ['rotate +90', 'rotate +180', 'diagonal translation']
networkName = "Inception"

# ----------Data Load-----------------
t1 = time.time()
IMG_SIZE = 75

classes = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6}
dl = DataLoader("C:\\Users\\tiwar\\Downloads\\FERG_DB_256",
                ".png",
                classes,
                IMG_SIZE,
                LIMIT)
if LOAD_DATA:
    images, labels = dl.load()
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=1)
    dl.save_train_test_split(X_train, X_test, y_train, y_test)
else:
    X_train, X_test, y_train, y_test = dl.load_train_test_split()
print("Data Load time: " + str(time.time() - t1))

# ---------CNN training---------------
t2 = time.time()
cnn = ConvolutionalNeuralNetworks(networkName, dl.datasetInfo)
cnn.create_model_architecture(X_train[0].shape)
statControllerNoRl = StatisticsController(classes)
if TRAIN_NETWORK:
    statControllerNoRl.trainingHistory = cnn.model.fit(X_train, dl.toOneHot(y_train), batch_size=20, epochs=400,
                                                       validation_split=0.2, callbacks=cnn.callbacks).history
    dl.save_training_history(statControllerNoRl.trainingHistory)
    dl.save_model(cnn.networkName, cnn.model)
else:
    dl.load_model_weights(networkName, cnn.model)
    statControllerNoRl.trainingHistory = dl.load_training_history()
print("CNN training time: " + str(time.time() - t2))

# ----------RL execution--------------
t4 = time.time()
q = QLearningModel()
statControllerRl = StatisticsController(classes, len(ACTION_NAMES))
verbose = True
for img, label in zip(X_test, y_test):
    no_lr_probabilities_vector = cnn.model.predict(NumpyImg2Tensor(img))
    predictedLabel = np.argmax(no_lr_probabilities_vector)
    statControllerNoRl.predictedLabels.append(predictedLabel)

    # article version:
    """
    if predictedLabel != label:
        q.perform_iterative_Q_learning(cnn, img, statControllerRl)
        optimal_action = q.choose_optimal_action()
        statControllerRl.updateOptimalActionsStats(optimal_action)
        corrected_img = q.apply_action(optimal_action, img)
        probabilities_vector = cnn.model.predict(NumpyImg2Tensor(corrected_img))
        statControllerRl.predictedLabels.append(np.argmax(probabilities_vector))
    else:
        statControllerRl.predictedLabels.append(predictedLabel)
    """

    # correct testing of article version:
    q.perform_iterative_Q_learning(cnn, img, statControllerRl)
    optimal_action = q.choose_optimal_action()
    statControllerRl.updateOptimalActionsStats(optimal_action)
    corrected_img = q.apply_action(optimal_action, img)

    probabilities_vector = cnn.model.predict(NumpyImg2Tensor(corrected_img))
    statControllerRl.predictedLabels.append(np.argmax(probabilities_vector))

    # best action, no RL version:
    """
    #optimal_action = q.action_space_search_choose_optimal(cnn, img, statControllerRl)
    #statControllerRl.updateOptimalActionsStats(optimal_action)
    #corrected_img = q.apply_action(optimal_action, img)
    #probabilities_vector = cnn.model.predict(NumpyImg2Tensor(corrected_img))
    #statControllerRl.predictedLabels.append(np.argmax(probabilities_vector))
    """

print("RL execution time: " + str(time.time() - t4))

plot_actions_stats(dl, networkName, ACTION_NAMES, statControllerRl.allActionsStats, "allActionsRL")
plot_actions_stats(dl, networkName, ACTION_NAMES, statControllerRl.optimalActionsStats, "optimalActionsRL")

conf_matrix_no_RL = confusion_matrix(y_test, statControllerNoRl.predictedLabels)
conf_matrix_RL = confusion_matrix(y_test, statControllerRl.predictedLabels)
plot_conf_matrix(dl, networkName, conf_matrix_no_RL, classes, "NoRL")
plot_conf_matrix(dl, networkName, conf_matrix_RL, classes, "RL")
plot_history(dl, networkName, statControllerNoRl.trainingHistory)

statControllerNoRl.f1Score = f1_score(y_test, statControllerNoRl.predictedLabels, average="macro")
statControllerNoRl.precision = precision_score(y_test, statControllerNoRl.predictedLabels, average="macro")
statControllerNoRl.recall = recall_score(y_test, statControllerNoRl.predictedLabels, average="macro")
statControllerNoRl.report = classification_report(y_test, statControllerNoRl.predictedLabels)
statControllerNoRl.accuracy = accuracy_score(y_test, statControllerNoRl.predictedLabels)

statControllerRl.f1Score = f1_score(y_test, statControllerRl.predictedLabels, average="macro")
statControllerRl.precision = precision_score(y_test, statControllerRl.predictedLabels, average="macro")
statControllerRl.recall = recall_score(y_test, statControllerRl.predictedLabels, average="macro")
statControllerRl.report = classification_report(y_test, statControllerRl.predictedLabels)
statControllerRl.accuracy = accuracy_score(y_test, statControllerRl.predictedLabels)

print_classification_details(statControllerNoRl)
print_classification_details(statControllerRl)
dl.save_details(statControllerNoRl, networkName, "NoRL")
dl.save_details(statControllerRl, networkName, "RL")


# In[14]:


import numpy as np
from random import randrange

from PIL import Image, ImageOps
from scipy.ndimage.interpolation import rotate


class QLearningModel:
    def __init__(self):
        self.alpha = 0.4
        self.gamma = 0.3
        self.angle1 = 90
        self.angle2 = 180
        self.angle3 = 10
        self.angle4 = -10
        # executing: self.actions[0](picture)
        self.actions = dict([(0, self.action_rotate_1), (1, self.action_rotate_2), (2, self.diagonal_translation)])
        self.states = [0, 1]
        self.tableQ = np.zeros((len(self.states), len(self.actions)))
        self.maxIter = len(self.actions) * 20

    def action_rotate_1(self, picture):
        return rotate(picture, self.angle1, reshape=False)

    def action_rotate_2(self, picture):
        return rotate(picture, self.angle2, reshape=False)

    def action_rotate_3(self, picture):
        return rotate(picture, self.angle3, reshape=False)

    def action_rotate_4(self, picture):
        return rotate(picture, self.angle4, reshape=False)

    def action_invariant(self, picture):
        return picture

    def diagonal_translation(self, picture):
        img = Image.fromarray(picture.astype('uint8'), 'RGB')
        w = int(img.size[0] * 0.75)
        h = int(img.size[1] * 0.75)
        border = (15, 15, img.size[0] - w - 15, img.size[1] - h - 15)
        img = img.resize((w, h), Image.ANTIALIAS)
        translated = ImageOps.expand(img, border=border, fill='black')
        return np.array(translated)

    def selectAction(self):
        return randrange(len(self.actions))

    def apply_action(self, action, img):
        return self.actions[action](img)

    def get_features_metric(self, features):
        return np.std(features)

    def get_reward(self, m1, m2):
        return np.sign(m2-m1)

    def define_state(self, reward):
        return 0 if reward > 0 else 1

    def update_tableQ(self, state, action, reward):
        self.tableQ[state][action] = self.tableQ[state][action] + self.alpha * (
            reward + self.gamma * max(self.tableQ[state]) - self.tableQ[state][action]
        )

    def action_space_search_choose_optimal(self, cnn, img, statsController):
        img_features = cnn.get_output_base_model(img)
        m1 = self.get_features_metric(img_features)
        optimal_action = 4
        for idx, action in enumerate(self.actions):
            statsController.updateAllActionStats(action)
            modified_img = self.apply_action(action, img)
            modified_img_features = cnn.get_output_base_model(modified_img)
            m2 = self.get_features_metric(modified_img_features)
            if m2 > m1:
                optimal_action = idx
        return optimal_action

    def perform_iterative_Q_learning(self, cnn, img, statsController):
        img_features = cnn.get_output_base_model(img)
        m1 = self.get_features_metric(img_features)
        for i in range(self.maxIter):
            action = self.selectAction()
            statsController.updateAllActionStats(action)
            modified_img = self.apply_action(action, img)
            modified_img_features = cnn.get_output_base_model(modified_img)
            m2 = self.get_features_metric(modified_img_features)
            reward = self.get_reward(m1, m2)
            state = self.define_state(reward)
            self.update_tableQ(state, action, reward)

    def choose_optimal_action(self):
        return np.where(self.tableQ == np.amax(self.tableQ))[1][0]


# In[15]:


import numpy as np


class StatisticsController:
    def __init__(self, classes, actions_cnt=0):
        self.classes = classes
        self.trainingHistory = None
        self.confMatrix = np.zeros((len(classes), len(classes)))
        self.optimalActionsStats = [0] * actions_cnt
        self.allActionsStats = [0] * actions_cnt
        self.predictedLabels = []
        self.recall = None
        self.precision = None
        self.f1Score = None
        self.report = None
        self.accuracy = None

    def updateOptimalActionsStats(self, action):
        self.optimalActionsStats[action] += 1

    def updateAllActionStats(self, action):
        self.allActionsStats[action] += 1


# In[ ]:




