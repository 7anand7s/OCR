import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from PIL import Image

def pad_img(img):
    old_h, old_w = img.shape[0], img.shape[1]

    # Pad the height.

    # If height is less than 512 then pad to 512
    if old_h < 512:
        to_pad = np.ones((512 - old_h, old_w)) * 255
        img = np.concatenate((img, to_pad))
        new_height = 512
    else:
        # If height >512 then pad to nearest 10.
        to_pad = np.ones((roundup(old_h) - old_h, old_w)) * 255
        img = np.concatenate((img, to_pad))
        new_height = roundup(old_h)

    # Pad the width.
    if old_w < 512:
        to_pad = np.ones((new_height, 512 - old_w)) * 255
        img = np.concatenate((img, to_pad), axis=1)
        new_width = 512
    else:
        to_pad = np.ones((new_height, roundup(old_w) - old_w)) * 255
        img = np.concatenate((img, to_pad), axis=1)
        new_width = roundup(old_w) - old_w
    return img


def pad_seg(img):
    old_h, old_w = img.shape[0], img.shape[1]

    # Pad the height.

    # If height is less than 512 then pad to 512
    if old_h < 512:
        to_pad = np.zeros((512 - old_h, old_w))
        img = np.concatenate((img, to_pad))
        new_height = 512
    else:
        # If height >512 then pad to nearest 10.
        to_pad = np.zeros((roundup(old_h) - old_h, old_w))
        img = np.concatenate((img, to_pad))
        new_height = roundup(old_h)

    # Pad the width.
    if old_w < 512:
        to_pad = np.zeros((new_height, 512 - old_w))
        img = np.concatenate((img, to_pad), axis=1)
        new_width = 512
    else:
        to_pad = np.zeros((new_height, roundup(old_w) - old_w))
        img = np.concatenate((img, to_pad), axis=1)
        new_width = roundup(old_w) - old_w
    return img


# This function helps in visualizing the image segmentation mask and image side by side using matplotlib subplots.
def visualize(img, seg_img):
    """
    Visualizes image
    """
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(seg_img, cmap='gray')
    plt.title('Segmented Image')
    plt.show()


def get_segmented_img(img, n_classes):
    """
    Loads in the segmented image and create suitable segmentation label.
    """
    seg_labels = np.zeros((512, 512, 1))
    # print('imggggg', img)
    img = cv2.resize(img, (512, 512))
    img = img[:, :, 0]
    cl_list = [0, 24]

    seg_labels[:, :, 0] = (img != 0).astype(int)

    return seg_labels


def preprocess_img(img, imgSize):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])
        print("Image None!")

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize, interpolation=cv2.INTER_CUBIC) # INTER_CUBIC interpolation best approximate the pixels image
                                                               # see this https://stackoverflow.com/a/57503843/7338066
    most_freq_pixel=find_dominant_color(Image.fromarray(img))
    target = np.ones([ht, wt]) * most_freq_pixel
    target[0:newSize[1], 0:newSize[0]] = img

    img = target

    return img


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def segment_into_lines(filename, model):
    line_img_array = []
    # Loading the image and performing thresholding on it and then resizing.
    img = cv2.imread(f'{filename}', 0)
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (512, 512))
    # Expanding the dimension to account for the batch dimension.
    img = np.expand_dims(img, axis=-1)
    # Expanding dimension along channel axis.
    img = np.expand_dims(img, axis=0)
    # Predict the segmentation mask.
    pred = model.predict(img)
    # Remove the batch and channel dimension for performing the binarization.
    pred = np.squeeze(np.squeeze(pred, axis=0), axis=-1)

    # Performing the binarization of the predicted mask for contour detection.
    coordinates = []
    img = cv2.normalize(src=pred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
    # Opening the original image to get the original dimension information.
    ori_img = cv2.imread(f'{filename}', 0)

    (H, W) = ori_img.shape[:2]
    (newW, newH) = (512, 512)
    rW = W / float(newW)
    rH = H / float(newH)

    # Contour detection and bouding box generation.
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(ori_img, (int(x*rW), int(y*rH)), (int((x+w)*rW),int((y+h)*rH)), (255,0,0), 1)
        coordinates.append((int(x * rW), int(y * rH), int((x + w) * rW), int((y + h) * rH)))
    # cv2.imwrite("output.jpg",ori_img)

    # Cropping the lines from the original image using the bouding boxes generated above.
    for i in range(len(coordinates) - 1, -1, -1):
        coors = coordinates[i]

        p_img = ori_img[coors[1]:coors[3], coors[0]:coors[2]].copy()

        line_img_array.append(p_img)

    return line_img_array


def segment_into_words(line_img, idx, model):
    """This function takes in the line image and line index returns word images and the reference
    of line they belong to."""
    img = pad_img(line_img)
    ori_img = img.copy()
    # ori_img=np.stack((ori_img,)*3, axis=-1)
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    img = cv2.resize(img, (512, 512))
    img = np.expand_dims(img, axis=-1)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    seg_pred = model.predict(img)
    seg_pred = np.squeeze(np.squeeze(seg_pred, axis=0), axis=-1)
    seg_pred = cv2.normalize(src=seg_pred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.threshold(seg_pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, seg_pred)
    contours, hier = cv2.findContours(seg_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    (H, W) = ori_img.shape[:2]
    (newW, newH) = (512, 512)
    rW = W / float(newW)
    rH = H / float(newH)

    coordinates = []

    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a white rectangle to visualize the bounding rect
        # cv2.rectangle(ori_img, (int(x*rW), int(y*rH)), (int((x+w)*rW),int((y+h)*rH)), (255,0,0), 1)
        coordinates.append((int(x * rW), int(y * rH), int((x + w) * rW), int((y + h) * rH)))

    coordinates = sort_word(coordinates)  # Sorting according to x-coordinates.
    word_counter = 0

    word_array = []
    line_indicator = []

    for (x1, y1, x2, y2) in coordinates:
        word_img = ori_img[y1:y2, x1:x2]
        word_img = preprocess_img(word_img, (128, 32))
        word_img = np.expand_dims(word_img, axis=-1)
        word_array.append(word_img)
        line_indicator.append(idx)

    return line_indicator, word_array


def sort_word(wordlist):
    wordlist.sort(key=lambda x: x[0])
    return wordlist


def find_dominant_color(image):
    # Resizing parameters
    width, height = 150, 150
    image = image.resize((width, height), resample=0)
    # Get colors from image object
    pixels = image.getcolors(width * height)
    # Sort them by count number(first element of tuple)
    sorted_pixels = sorted(pixels, key=lambda t: t[0])
    # Get the most frequent color
    dominant_color = sorted_pixels[-1][1]
    return dominant_color
