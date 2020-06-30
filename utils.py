"""
Author: Yufei Chen
Mail: yfchen@sei.xjtu.edu.cn
"""

import os
import sys
import cv2
import numpy as np

def imgLoader(imgPath, color_flag=cv2.IMREAD_COLOR):
    """
    Read the image and from imagePath using OpenCV.

    Args:
        imgPath: Where the image is saved.
        color_flg: Specify the way the image should be read.

    Returns:
        img: The image data load from imgPath.
             A numpy matrix shaped in (width, height, channels)
    """
    try:
        img = cv2.imread(imgPath, color_flag)
        height, width, *channel = img.shape
        if not channel:
            img = img.reshape((height, width, 1))
        return img
    except:
        print('Fail to load the image %s' %imgPath)

def imgSaver(imgPath, img):
    """
    Save the image at imagePath using OpenCV.

    Args:
        imgPath: Where the image to save.
        img: The image data to save.
             A numpy matrix shaped in (width, height, channels)
    """
    try:
        img = cv2.imwrite(imgPath, img)
        return img
    except:
        print('Fail to save the image as %s' %imgPath)

def color_shift(sourceImg, targetImg):
    source = cv2.resize(sourceImg, (targetImg.shape[0],targetImg.shape[1]))
    source = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    target = cv2.cvtColor(targetImg, cv2.COLOR_BGR2HSV)
    target[:,:,0:2] = source[:, :, 0:2]
    target_shifted = cv2.cvtColor(target, cv2.COLOR_HSV2BGR)
    return target_shifted

def darknet_resize(img, outShape):
    w, h = outShape
    if len(img.shape) == 2:
        img = np.reshape(img, (img.shape[0], img.shape[1],1))
    img_h, img_w, img_c = img.shape
    w_scale = (img_w - 1.0) / (w - 1.0)
    h_scale = (img_h - 1.0) / (h - 1.0)
    new_img = np.zeros((h,w,img_c))
    part = np.zeros((img_h,w,img_c))
    for k in range(img_c):
        for r in range(img_h):
            for c in range(w):
                val = 0
                if c == w-1 or img_w == 1:
                    val = img[r, img_w-1, k]
                else:
                    sx = c * w_scale
                    ix = int(sx)
                    dx = 1.0 * (sx - ix)
                    val = (1-dx) * img[r,ix,k] + dx * img[r,ix+1,k]
                part[r,c,k] = val

    for k in range(img_c):
        for r in range(h):
            sy = r*h_scale
            iy = int(sy)
            dy = 1.0 * (sy-iy)
            for c in range(w):
                val = (1-dy) * part[iy,c,k]
                new_img[r,c,k] = val
            if r == h-1 or img_h == 1:
                continue
            for c in range(w):
                val = dy * part[iy+1,c,k]
                new_img[r,c,k] = new_img[r,c,k] + val
    new_img = np.uint8(new_img)
    return new_img

def test():
    pass

if __name__ == '__main__':
    a = cv2.imread('attack_yolo_320.jpg')
    b = darknet_resize(a, (320, 320))
    cv2.imwrite("resized_darknet.jpg", b)
