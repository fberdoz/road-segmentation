import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from scipy import ndimage, signal, stats
from helpers import *

def PlotAugmentedData(imgs, gt_imgs, id_img=0):
    """ 
    Plot one image (id_img) and its corresponding rotations and symmetries.
    The angles for the rotations are [0, 45, 90, 135, 180, 225, 270, 315].
    """
    
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
    """ Add three chanels to the original images. (Grey level, vertical edge and horizontal edges)"""
    
 
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
    """ Create a 3x3 grid of the imput image by mirroring it at the boundaries"""

    top_and_lower_row = np.concatenate((np.flip(np.flip(img,0),1), np.flip(img,0), np.flip(np.flip(img,0),1)), axis=1)
    mid_row = np.concatenate((np.flip(img,1), img, np.flip(img,1)), 1)
    
    ext_img = np.concatenate((top_and_lower_row, mid_row, top_and_lower_row), 0)
       
    return ext_img

def BuildRotatedImage(img, degree):
    """ Return the same image rataded by <degree> degrees. The corners are filled using mirrored boundaries. """
    
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
    """
    Augments the data by rotating the image with the angles given in the list <angles>.
    If <sym> is true, it also augments the the data with the images obtained by performing a
    y axis symmetry.
    
    The augmented data will present itself as follows: 
        
        [angles[1] rotations, ..., angles[end] rotations,
        Y axis symmetry of the angles[1] rotations, ..., Y axis symmetry of the angles[end] rotations]
 
    """
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
        
    return aug_imgs, aug_gt_imgs

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