import os
import glob
import airsim
import numpy as np
import pandas as pd

from scipy import ndimage


# read image data
def read_images(folder):
    print("Loading images")
    files_left = sorted(glob.glob(folder + "/img_Left_5_*.png"))
    images_left = np.array([ndimage.imread(file, mode="L") for file in files_left], np.uint8)
    print("Loading images 30%")
    files_center = sorted(glob.glob(folder + "/img_Center_5_*.png"))
    images_center = np.array([ndimage.imread(file, mode="L") for file in files_center], np.uint8)
    print("Loading images 60%")
    files_right = sorted(glob.glob(folder + "/img_Right_5_*.png"))
    images_right = np.array([ndimage.imread(file, mode="L") for file in files_right], np.uint8)
    return images_left[:, :, :, np.newaxis], images_center[:, :, :, np.newaxis], images_right[:, :, :, np.newaxis]


# read text information
def read_text(file):
    df = pd.read_csv(file, sep="\t")

    speed = df[["Speed"]].values
    control = df[["Throttle", "Steering"]].values
    control[:, 0] -= df[["Brake"]].values[:, 0]
    return speed, control


# load data folder
def load_folder(folder, ground_id=95):
    # load data
    speed, control = read_text("./{}/airsim_rec.txt".format(folder))
    images_left, images_center, images_right = read_images("./{}/images".format(folder))

    # preprocess data
    speed /= 20.
    images_left = (1 * (images_left == ground_id)).astype(np.uint8)
    images_center = (1 * (images_center == ground_id)).astype(np.uint8)
    images_right = (1 * (images_right == ground_id)).astype(np.uint8)

    # image history
    speed = speed[2:]
    control = control[2:]
    images_left = np.concatenate((images_left, np.roll(images_center, 1, axis=0), np.roll(images_center, 2, axis=0)), axis=-1)[2:]
    images_right = np.concatenate((images_right, np.roll(images_center, 1, axis=0), np.roll(images_center, 2, axis=0)), axis=-1)[2:]
    images_center = np.concatenate((images_center, np.roll(images_center, 1, axis=0), np.roll(images_center, 2, axis=0)), axis=-1)[2:]

    return speed, control, images_left, images_center, images_right


# get data live
image_translate = np.array([299, 587, 114]) / 1000.
def data_live(client, gound_id=105):
    # gather and preprocess car data
    speed = client.getCarState().speed / 20.

    # preprocess image
    image_current = client.simGetImages([airsim.ImageRequest("Center", airsim.ImageType.Segmentation, False, False)])[0]
    image_1d = np.fromstring(image_current.image_data_uint8, dtype=np.uint8)

    if image_current.height != 0 and image_current.width != 0:
        image_current = image_1d.reshape(image_current.height, image_current.width, 3)
        image_current = np.einsum("xyz, z -> xy", image_current, image_translate).astype(np.uint8)
        image = (1 * (image_current == gound_id)).astype(np.uint8)
        return speed, image

    return False