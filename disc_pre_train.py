import random
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import Model
from tensorflow.keras import layers
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
from ISR.models import Discriminator
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "galaxy_zoo"
CATEGORIES = ["individuals", "merged"]

FOLDER = []
for i in [2, 3, 4]:
    for c in CATEGORIES:
        for t in ["valid", "train"]:
            FOLDER.append(c+"_"+str(i)+"blend_"+t)

training_data = []

for f in FOLDER:
    path = os.path.join(DATADIR, f)
    class_num = 0
    if CATEGORIES[1] in f:
        class_num = 1

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(
                path, img), cv2.IMREAD_UNCHANGED)
            new_array = cv2.resize(img_array, (80, 80))

            training_data.append([new_array, class_num])
        except Exception as e:
            pass

random.shuffle(training_data)

batch_size = 12
IMG_SIZE = 80

discr = Discriminator(patch_size=IMG_SIZE, kernel_size=3)
model = Model(discr.model.input, discr.model.output)
discr_out_shape = list(model.outputs[0].shape)[1:4]

X = []
y = []
z = None
for features, label in training_data:
    X.append(features)

    if label == 1:
        # z=[0]*400
        z = 0
    else:
        # z=[1]*400
        z = 1

    y.append(z)




X = np.array(X)
y = np.array(y)

X = np.array(X).reshape(-1, 80, 80, 3)
y = np.array(y).reshape(-1, 1)

model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

filepath = "model/21/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, save_weights_only=True, mode='max')
tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

model.fit(X, y,
                batch_size=batch_size,
                epochs=10000,
                validation_split=0.3,
                callbacks=[tensorboard, checkpoint])