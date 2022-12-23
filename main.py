# Importing required libraries.
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import gdown
import random
from keras.callbacks import ModelCheckpoint
from utils import get_segmented_img, preprocess_img, visualize, segment_into_lines, segment_into_words, pad_img, pad_seg
from RecognizeWord import recognize_words
from model import unet

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))#

# %tensorflow_version 1.x
# gdown.download("https://drive.google.com/uc?id=155wulmctFNkAMYFp770XLCWm1FcYEsfy", 'WordSegData2.zip' )
# exit()

# image_list = os.listdir(r'C:/Users/77ana/PycharmProjects/ocr/content/Dataset1/img/')
# image_list = [filename.split(".")[0] for filename in image_list]


def batch_generator(filelist, n_classes, batch_size):
    while True:
        X = []
        Y = []
        for i in range(batch_size):
            fn = random.choice(filelist)
            img = cv2.imread(f'C:/Users/77ana/PycharmProjects/ocr/content/Dataset1/img/{fn}.jpg', 0)
            img = pad_img(img)
            ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

            img = cv2.resize(img, (512, 512))
            img = np.expand_dims(img, axis=-1)
            # img = np.stack((img,)*3, axis=-1)
            img = img / 255

            seg = cv2.imread(f'C:/Users/77ana/PycharmProjects/ocr/content/Dataset1/mask/{fn}_mask.png', 0)
            seg = pad_seg(seg)
            seg = cv2.resize(seg, (512, 512))
            seg = np.stack((seg,) * 3, axis=-1)
            seg = get_segmented_img(seg, n_classes)

            X.append(img)
            Y.append(seg)
        yield np.array(X), np.array(Y)


# model = FCN(n_classes=2,
#  input_height=320,
#  input_width=320)
LS_model = unet()
LS_model.load_weights('C:/Users/77ana/PycharmProjects/ocr/LS_weights00000001.h5')

WS_model = unet()
WS_model.load_weights('C:/Users/77ana/PycharmProjects/ocr/WS_weights00000001.h5')
# model.summary()

# Let’s split the dataset into training and test set. I decided to use 70% data for training and 30% for testing.
# random.shuffle(image_list)
# file_train = image_list[0:int(0.75 * len(image_list))]
# file_test = image_list[int(0.75 * len(image_list)):]

# Let’s train the model.
# mc = ModelCheckpoint('weights00000001.h5',
#                      save_weights_only=True, period=1)
# model.fit_generator(batch_generator(file_train, 2, 2), epochs=3, steps_per_epoch=1000,
#                     validation_data=batch_generator(file_test, 2, 2),
#                     validation_steps=400, callbacks=[mc], shuffle=1)

# -------------------------------------
# Open image and segment into lines
line_img_array = segment_into_lines(r'C:\Users\77ana\PycharmProjects\ocr\content\test_img.JPG', LS_model)
# print(line_img_array)

# Creating lists to store the line indexes,words list.
full_index_indicator = []
all_words_list = []
# Variable to count the total no of lines in page.
len_line_arr = 0

# Segment the lines into words and store as arrays.
for idx, im in enumerate(line_img_array):
    line_indicator, word_array = segment_into_words(im, idx, WS_model)
    for k in range(len(word_array)):
        full_index_indicator.append(line_indicator[k])
        all_words_list.append(word_array[k])
    len_line_arr += 1

all_words_list = np.array(all_words_list)

# Perform the recognition on list of list of words.
recognize_words(full_index_indicator, all_words_list, len_line_arr)