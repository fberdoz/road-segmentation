import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import pickle
from PIL import Image
from scipy import ndimage, signal
import re

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
    

foreground_threshold = (
    0.25
)  # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i : i + patch_size, j : j + patch_size]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, "w") as f:
        f.write("id,prediction\n")
        for fn in image_filenames:
            f.writelines("{}\n".format(s) for s in mask_to_submission_strings(fn))


    if __name__ == "__main__":
        submission_filename = "dummy_submission.csv"
        image_filenames = []
        for i in range(1, 51):
            image_filename = "training/groundtruth/satImage_" + "%.3d" % i + ".png"
            print(image_filename)
            image_filenames.append(image_filename)
            masks_to_submission(submission_filename, *image_filenames)

# Convert an array of binary labels to a uint8
def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg


def reconstruct_from_labels(image_id, submissionfile):
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open(submissionfile)
    lines = f.readlines()
    image_id_str = "%.3d_" % image_id
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(",")
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split("_")
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j + w, imgwidth)
        ie = min(i + h, imgheight)
        if prediction == 0:
            adata = np.zeros((w, h))
        else:
            adata = np.ones((w, h))

        im[j:je, i:ie] = binary_to_uint8(adata)

    Image.fromarray(im).save("prediction_" + "%.3d" % image_id + ".png")

    return im


