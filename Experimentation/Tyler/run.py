import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import pickle
import cv2
import tensorflow as tf
from tensorflow.python import keras
from skimage import img_as_float
from PIL import Image
from scipy import ndimage, signal

# TRAINING HELPERS
def load_image(infilename):
    data = mpimg.imread(infilename)
    return data
def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg
# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches
def TruePositiveRate(X, Y, model):
    # Predict on the training set
    Y_pred = np.argmax(model.predict(X),axis=1)
    # Get non-zeros in prediction and grountruth arrays
    Y_predn = np.nonzero(Y_pred)[0]
    Yn = np.nonzero(Y)[0]
    TPR = len(list(set(Yn) & set(Y_predn))) / float(len(Yn))
    return TPR
def PlotAugmentedData(imgs, gt_imgs, id_img=0):
    aug_imgs, aug_gt_imgs = DataAugmentation([imgs[id_img]], [gt_imgs[id_img]], [0, 45, 90, 135, 180, 225, 270, 315], sym=True)
    nimgs = int(len(aug_imgs)/16)

    titles = ["Original image", 
              "Rotated image (45°)", 
              "Rotated image (90°)",
              "Rotated image (135°)", 
              "Rotated image (180°)", 
              "Rotated image (225°)", 
              "Rotated image (270°)",
              "Rotated image (315°)", 
              "Y axis sym. of the \n original version",
              "Y axis sym. of the \n rotated image (45°)", 
              "Y axis sym. of the \n rotated image (90°)",
              "Y axis sym. of the \n rotated image (135°)", 
              "Y axis sym. of the \n rotated image (180°)", 
              "Y axis sym. of the \n rotated image (225°)",
             "Y axis sym. of the \n rotated image (270°)",
             "Y axis sym. of the \n rotated image (305°)"]
    
    fig_augData, axs = plt.subplots(4, 4, figsize=(16, 12))
    for i in range(16):
        print
        axs[int(i/4), (i % 4)].imshow(concatenate_images(aug_imgs[i], aug_gt_imgs[i]), cmap='Greys_r')
        axs[int(i/4), (i % 4)].set_title(files[id_img] + "\n" + titles[i])  
            
def ChannelAugmentation(imgs):
    new_imgs = []
    #Sobel filters masks
    V_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    H_kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    # Performing the correlation on each image
    for i, img in enumerate(imgs):
        Grey_level = img.mean(axis=2)
        V_Edges = signal.correlate2d(Grey_level, V_kernel, mode='same', boundary='symm')
        H_Edges = signal.correlate2d(Grey_level, H_kernel, mode='same', boundary='symm')
        new_imgs += [np.dstack((img, Grey_level, V_Edges, H_Edges))]
    return new_imgs
def BuildExtendedImage(img):
    top_and_lower_row = np.concatenate((np.flip(np.flip(img,0),1), np.flip(img,0), np.flip(np.flip(img,0),1)), axis=1)
    mid_row = np.concatenate((np.flip(img,1), img, np.flip(img,1)), 1)
    ext_img = np.concatenate((top_and_lower_row, mid_row, top_and_lower_row), 0)
    return ext_img

def BuildRotatedImage(img, degree):
    # Improving performance using existing functions for specific angles
    if (degree==0):
        return img
    elif (degree==90):
        return np.rot90(img)
    elif (degree==180):
        return np.rot90(np.rot90(img))
    elif (degree==270):
        return np.rot90(np.rot90(np.rot90(img)))
    else:
        h = img.shape[0]
        w = img.shape[1]
        # Extend and rotate the image
        ext_img = BuildExtendedImage(img)
        rot_img = ndimage.rotate(ext_img, degree, reshape=False)
        # Taking care of nummerical accuracies (not sure where they come from)
        rot_img[rot_img<0] = 0.0
        rot_img[rot_img>1] = 1.0
        # Crop the image
        if (len(img.shape) > 2):
            rot_img = rot_img[h:2*h, w:2*w, :]
        else:
            rot_img = rot_img[h:2*h, w:2*w]
        return rot_img
def DataAugmentation(imgs, gt_imgs, angles, sym=True):
    n = len(imgs)
    # Creating the augmented version of the images.
    aug_imgs = []
    aug_gt_imgs = []
    # Rotating the images and adding them to the augmented data list
    for theta in angles:
        print("Augmenting the data with the images rotated by", theta , "deg.")
        aug_imgs += [BuildRotatedImage(imgs[i], theta) for i in range(n)]
        aug_gt_imgs += [BuildRotatedImage(gt_imgs[i], theta) for i in range(n)]     
    # Y symmetry of the images and adding them to the augmented dat list
    if (sym):
        print("Augmenting the data with the symmetries")
        n_tmp = len(aug_imgs)
        aug_imgs += [np.flip(aug_imgs[i],1) for i in range(n_tmp)]
        aug_gt_imgs += [np.flip(aug_gt_imgs[i],1) for i in range(n_tmp)]
    return np.array(aug_imgs), np.array(aug_gt_imgs)
def ExtractPatch(imgs, gt_imgs, patch_size=16):
    """ Extract patches of size patch_size from the input images.  """ 
    n = len(imgs)
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]
    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    return img_patches, gt_patches
def ExtractPatchTest(imgs, patch_size=16):
    """ Extract patches of size patch_size from the input images.  """ 
    n = len(imgs)
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    return img_patches
def extract_features(img):
    th_down = 0.31 #80/255
    th_up = 0.78   #200/255
    road = 0;
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))  
    if ((feat_m[3] > th_down)&(feat_m[3] < th_up)): 
        road = 1;
    road = np.array([road])
    feat_p = stats.mode(img[:,:,3], axis=None)[0]
    features = np.concatenate((feat_m, feat_v, feat_p, road))
    return features
# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    th_down = 0.31 #80/255
    th_up = 0.78   #200/255
    road = 0;
    feat_m = np.mean(img)
    if ((feat_m[3] > th_down)&(feat_m[3] < th_up)): 
        road = 1;
    feat_v = np.var(img)
    feat_p = stats.mode(img, axis=None)[0]
    features = np.append(feat_m, feat_v, feat_p, road)
    return features
# Extract features for a given image
def extract_img_features(filename,patch_size):
    img_raw = load_image(filename)
    [img] = ChannelAugmentation([img_raw])
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([extract_features(img_patches[i]) for i in range(len(img_patches))])
    return X, img, img_raw
def build_poly(X, degree, interaction_only=False):
    """polynomial basis functions for input data X, for j=0 up to j=degree. """
    poly = PolynomialFeatures(degree, interaction_only)
    return poly.fit_transform(X)
def value_to_class(v, foreground_threshold):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0   
def PrintFeatureStatistics(X, Y):
    print('There are ' + str(X.shape[0]) + ' data points')
    print('Each data point has ' + str(X.shape[1]) + " features")
    #print('Number of classes = ' + str(np.max(Y)))  #TODO: fix, length(unique(Y)) 
    print('Number of classes = ' + str(len(np.unique(Y))))
    Y0 = [i for i, j in enumerate(Y) if j == 0]
    Y1 = [i for i, j in enumerate(Y) if j == 1]
    print('Class 0 (background): ' + str(len(Y0)) + ' samples')
    print('Class 1 (signal): ' + str(len(Y1)) + ' samples')
def NormalizeFeatures(X):
    """Normalize X which must have shape (num_data_points,num_features)"""
    m = np.mean(X,axis=0)
    s = np.std(X,axis=0)   
    return (X-m)/s
patch_size = 32
def GetPixelPatchArray(imgs,gt_imgs,patch_size=32):
    sz = imgs.shape
    if (sz[1]%patch_size)!=0:
        print("Error, use different patch_size")
        raise ImplementedError
    patch_array = np.zeros([sz[0]*int(sz[1]/patch_size)**2,patch_size,patch_size,3])
    gt_patch_array = np.zeros([sz[0]*int(sz[1]/patch_size)**2,patch_size,patch_size,2])
    
    it = 0
    for img,gt_img in zip(imgs,gt_imgs):
        for i in range(int(sz[1]/patch_size)):
            for j in range(int(sz[1]/patch_size)):
                patch_array[it] = img[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
                gt_patch_array[it] = gt_img[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
                it += 1
    del imgs, gt_imgs
    return patch_array, gt_patch_array
def GetPixelPatchImage(img,patch_size=32):
    sz = img.shape
    if (sz[0]%patch_size)!=0:
        print("Error, use different patch_size")
        raise ImplementedError
    patch_img = np.zeros([int(sz[1]/patch_size)**2,patch_size,patch_size,3])
    for i in range(int(sz[1]/patch_size)):
        for j in range(int(sz[1]/patch_size)):
            patch_img[it] = img[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
    return patch_img

def GetPatchImg(img,gt_img,patch_size=16):
    """Returns a 3D tensor of patches. 0th and 1st axes are image coords and 2nd is channels."""
    sz = img.shape
    img = tf.keras.utils.normalize(np.array([img]))[0]
    if (sz[0]%patch_size)!=0:
        print("Error, use different patch_size")
        raise ImplementedError
    patchimg = np.zeros([int(sz[0]/patch_size),int(sz[1]/patch_size),12])
    gt_patchimg = np.zeros([int(sz[0]/patch_size),int(sz[1]/patch_size),2])
    for i in range(patchimg.shape[0]):
        for j in range(patchimg.shape[1]):
            pixelregion = img[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
            gtregion = gt_img[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
            patchimg[i,j,:] = np.hstack((np.mean(pixelregion,axis=(0,1)),np.var(pixelregion,axis=(0,1))))
            if np.mean(gtregion,axis=(0,1))>0.25:
                gt_patchimg[i,j,1]=1
            else:
                gt_patchimg[i,j,0]=1
    
    return patchimg,gt_patchimg

# TRAINING DATA LOADING
root_dir = '../Project-ML/Tyler/training/'
gt_dir = root_dir + 'groundtruth/'
image_dir = root_dir + 'images/'
n = 100
imgs = []
gt_imgs = [] 

for i in range(n):
    if i+1<10:
        number = '00'+str(i+1)
    elif i+1<100:
        number = '0'+str(i+1)
    else:
        number = str(i+1)
    imgs.append(cv2.resize(load_image(image_dir+"satImage_"+number+".png"),(608,608)))
    gt_imgs.append(cv2.resize(load_image(gt_dir+"satImage_"+number+".png"),(608,608)))

# DATA MANIPULATION
# Image equalization
for i in range(n):
    uint8 = img_float_to_uint8(imgs[i])
    r,g,b = cv2.split(uint8)
    uint8_equalized = cv2.merge((cv2.equalizeHist(r), cv2.equalizeHist(g), cv2.equalizeHist(b)))
    imgs[i] = (img_as_float(uint8_equalized));
# Edge Enhancement
for i, img in enumerate(imgs):
    imgs[i] = ChannelAugmentation([img])[0]

patch_size=16
img_array = np.zeros([len(imgs),int(imgs[0].shape[0]/patch_size),int(imgs[0].shape[1]/patch_size),12])
gt_img_array = np.zeros([len(imgs),int(imgs[0].shape[0]/patch_size),int(imgs[0].shape[1]/patch_size),2])
for i in range(n):
    img_array[i],gt_img_array[i] = GetPatchImg(imgs[i],gt_imgs[i],patch_size)
del imgs, gt_imgs

angles = [0,90,180,270]
y_train = np.zeros([200*len(angles),img_array.shape[1],img_array.shape[2],2])
x_train, y_train = DataAugmentation(img_array, gt_img_array, angles, True)
plt.imshow(x_train[0,:,:,:3])
del img_array, gt_img_array

from tensorflow.python.keras.models import load_model
model = load_model('model16-10.h5')
#model = keras.Sequential()
#model.add(tf.keras.layers.Dense(100,activation='relu',input_shape=x_train.shape[1:4]))
#model.add(tf.keras.layers.Dense(100,activation='relu'))
#model.add(tf.keras.layers.Dense(100,activation='relu'))
#model.add(tf.keras.layers.Dense(100,activation='sigmoid'))
#model.add(tf.keras.layers.Dense(2,activation='softmax'))
#model.summary()
#model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
#TRAINING
model.save('model16-10.h5')
i = 0
while i<100:
    model.fit(x_train,y_train,epochs=5,validation_split=0.2,verbose=2)
    i=i+5
    print(i)
    model.save('model16-10.h5')

# SUBMISSION HELPERS
from datetime import datetime
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
# TESTING DATA LOADING
root_dir = "../Project-ML/Tyler/test/test_"
n = np.arange(0,51)[1:]

test_imgs = [load_image(root_dir + str(i) + "/test_" + str(i) + ".png") for i in n]
print("Loaded test set of 50 images")
print("Their shapes are",test_imgs[0].shape)

# DATA MANIPULATION
# Image equalization
n=50
for i in range(n):
    uint8 = img_float_to_uint8(test_imgs[i])
    r,g,b = cv2.split(uint8)
    uint8_equalized = cv2.merge((cv2.equalizeHist(r), cv2.equalizeHist(g), cv2.equalizeHist(b)))
    test_imgs[i] = (img_as_float(uint8_equalized));
# Edge Enhancement
for i, img in enumerate(test_imgs):
    test_imgs[i] = ChannelAugmentation([img])[0]

prediction_filenames = []
for i,img in enumerate(test_imgs):
    patch_img = GetTestPatch(img,16)
    gt_patch = np.argmax(model.predict(np.array([patch_img]))[0],axis=-1)
    y_pred = TestPatchConversion(gt_patch,patch_size=16)
    if i+1<10:
        image_string = "00"+str(i+1)
    else:
        image_string = "0"+str(i+1)
    prediction_filenames += ["Submission/image_"+image_string+".png"]    
    # Saving the masks in the preddir folder
    Image.fromarray(binary_to_uint8(y_pred)).save(prediction_filenames[i])  

now = datetime.now()
dt_string = now.strftime("%H_%M__%d_%m")
submission_filename ="Submission/submission_" + dt_string + ".csv"
    
# Create submission
masks_to_submission(submission_filename, prediction_filenames)
