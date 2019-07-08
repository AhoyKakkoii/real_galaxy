from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
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

if 'total.pickle' not in os.listdir():
	FOLDER = []
	for i in [2,3,4]:
		for c in CATEGORIES:
			for t in ["valid", "train"]:
				FOLDER.append(c+"_"+str(i)+"blend_"+t)

	training_data = []

	for f in FOLDER:
		path = os.path.join(DATADIR, f)
		class_num=0
		if CATEGORIES[1] in f:
			class_num=1

		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_UNCHANGED)
				new_array = cv2.resize(img_array, (80, 80))

				training_data.append([new_array, class_num])
				print(img, class_num)
			except Exception as e:
				pass
	print(len(training_data))

	pickle_out = open("total.pickle","wb")
	pickle.dump(training_data, pickle_out)
	pickle_out.close()

pickle_in = open("total.pickle","rb")
training_data = pickle.load(pickle_in)


if 'x.pickle' not in os.listdir() and 'y.pickle' not in os.listdir():
	import random
	random.shuffle(training_data)
	x = []
	y = []
	z = None
	for features, label in training_data:
		x.append(features)
		
		if label == 1:
			#z=[0]*400
			z=0
		else:
			#z=[1]*400
			z=1

		print(len(y), len(training_data))
		y.append(z)

	x = np.array(x).reshape(-1,80,80,3)
	# y = np.array(y).reshape(-1,20,20,1)

	pickle_out = open("x.pickle","wb")
	pickle.dump(x, pickle_out)
	pickle_out.close()

	pickle_out = open("y.pickle","wb")
	pickle.dump(y, pickle_out)
	pickle_out.close()


pickle_in = open("x.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

IMG_SIZE=80


from ISR.models import Discriminator
from keras.layers import Flatten, Dense
from keras.models import Model

discr = Discriminator(patch_size=IMG_SIZE, kernel_size=3)
m = Flatten()(discr.model.output)
m = Dense(1, activation="sigmoid")(m)
model = Model(discr.model.input,m)

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
import tensorflow as tf


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

filepath = "model/21/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

model.fit(X, y,
          batch_size=2,
          epochs=1,
          validation_split=0.3,       
          callbacks=[tensorboard, checkpoint])

def get_best_21weights():
	maxx=0
	for f in os.listdir('model/21'):
		n = int(f.split('.')[1].split('-')[-1])
		print(n, maxx)
		maxx = n if maxx < n else maxx

	for f in os.listdir('model/21'):
		n = int(f.split('.')[1].split('-')[-1])
		if maxx == n:
			return f

weights_name = get_best_21weights()
model.load_weights('model/21/'+weights_name)
new_weights_path = 'model/24/'+weights_name

for f in os.listdir('model/24'):
	os.remove('model/24/'+f)

Model(inputs=model.input, outputs=model.layers[-3].output).save_weights(new_weights_path)
