#Dataset: The data having the images of Food and Groceries products.The train data has 3215 product images. The test data has 1732 product images. The data is taken from analytics vidya.
#The numeric dataset has the two columns which has the image id and labels.

## import libaries
import pandas as pd
import numpy as np
import cv2
import os, sys
from tqdm import tqdm
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


## load data
train = pd.read_csv('/home/amar/image_detection_data_set/train.csv')
test = pd.read_csv('/home/amar/image_detection_data_set/test.csv')
## set path for images
TRAIN_PATH = 'train_img/'
TEST_PATH = 'test_img/'

# function to read image
def read_img2(img_path2):
    img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
    img2 = cv2.resize(img2, (128,128))
    return img2
# load train and test data
#train_img2, test_img2 = [],[]
for img_path2 in tqdm(train['image_id'].values):
    train_img2.append(read_img2(TRAIN_PATH + img_path2 + '.png'))

for img_path2 in tqdm(test['image_id'].values):
    test_img2.append(read_img2(TEST_PATH + img_path2 + '.png'))

# normalize images
x_train2 = np.array(train_img2, np.float32) / 255.
x_test = np.array(test_img2, np.float32) / 255.

# target variable - encoding numeric value. 
label_list2 = train['label'].tolist()
Y_train2 = {k:v+1 for v,k in enumerate(set(label_list2))}
y_train2 = [Y_train2[k] for k in label_list2]   
y_train2 = np.array(y_train2)
y_train2 = to_categorical(y_train2)
#Transfer learning with Inception V3 
base_model2 = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

## set model2 architechture 
add_model2 = Sequential()
add_model2.add(Flatten(input_shape=base_model2.output_shape[1:]))
add_model2.add(Dense(256, activation='relu'))
add_model2.add(Dense(y_train2.shape[1], activation='softmax'))

model2 = Model(inputs=base_model2.input, outputs=add_model2(base_model2.output))
#Adam's optimizer, LR=Pow(10,-4), Accuracy Metric
model2.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

#Set batch_size, epochs
batch_size = 28 # tune it
epochs = 10 # increase it

train_datagen2 = ImageDataGenerator(
        rotation_range=30, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True)
train_datagen2.fit(x_train2)


history2 = model2.fit_generator(
    train_datagen2.flow(x_train2, y_train2, batch_size=batch_size),
    steps_per_epoch=x_train2.shape[0] // batch_size,
    epochs=epochs,
    callbacks=[ModelCheckpoint('Inception-transferlearning.model2', monitor='val_acc', save_best_only=True)]
)

#predict test data
predictions2 = model2.predict(x_test)
# get labels
predictions2 = np.argmax(predictions2, axis=1)
rev_y2 = {v:k for k,v in Y_train2.items()}
pred_labels2 = [rev_y2[k] for k in predictions2]

#Accuracy of 92 percent.
