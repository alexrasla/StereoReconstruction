from curses import window
from curses.ascii import LF
import math
from operator import index
from turtle import left, right
from matplotlib.path import Path
import cv2
import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("--left")
parser.add_argument("--right")
parser.add_argument("--scale")
parser.add_argument("--out")
args = parser.parse_args()

class Config:
    FOCAL_LENGTH = 3740 #pixels
    BASELINE = 160 #mm
    WINDOW_SIZE = 10
    WINDOW_CENTER = (WINDOW_SIZE - 1)//2

def stereo_reconstruction(left_img_path, right_img_path):
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    assert(left_img.shape == right_img.shape)
    
    stereo_img = np.zeros((left_img.shape[0], left_img.shape[1]), dtype=np.uint8)
    
    left_windows = np.lib.stride_tricks.sliding_window_view(left_img, (Config.WINDOW_SIZE, Config.WINDOW_SIZE, 3))
    right_windows = np.lib.stride_tricks.sliding_window_view(right_img, (Config.WINDOW_SIZE, Config.WINDOW_SIZE, 3))    

    for scanline_idx in range(left_windows.shape[0]):
        
        # print(scanline_idx)
        indexes = ssd(left_windows[scanline_idx], right_windows[scanline_idx])#what we want
        stereo_img[scanline_idx, :indexes.shape[0]] = (Config.FOCAL_LENGTH * Config.BASELINE) / (indexes - np.arange(indexes.shape[0]))

        if scanline_idx%100 == 0:
            print(scanline_idx)
        #     cv2.imshow('stereo', stereo_img)
        #     cv2.waitKey(0)
    return stereo_img
                

def ssd(window1, window2):
    '''
    window1 (left) shape == (m,1,n,n,3)
    window2 (right) shape == (m,1,n,n,3)
    '''
    
    #flatten each into 1262 by 1xnxnx3
    
    window1_flat = np.reshape(window1, (window1.shape[0], -1))
    window2_flat = np.reshape(window2, (window2.shape[0], -1))
    
    # for left in window1_flat:
    #     #init array
    #     arr = []
    #     for right in window2_flat:
    #         #compare each left to the right
    #         ssd_t = np.sum((left - right)**2) #score
    #         #add to array
    #         arr.append(ssd_t)
    #     # print(arr)
    #     print(np.max(np.array(arr))) #"best score" for right at left window
    #     break
    
    
    ssd = np.sum((window1_flat[:, None] - window2_flat)**2, axis=-1)
    
    # print(np.argmax(ssd[0])) #max 
    # print(np.argmax(ssd, axis=1))

    return np.argmin(ssd, axis=1) #each index is left pixel, each index value is right pixel

    # for left_window_idx in range(len(window1)):
        
    #     std1 = np.std(window1, axis=(1, 2, 3, 4))#np.std(window1[left_window_idx])
    #     std2 = np.std(window2, axis=(1, 2, 3, 4))
    #     avg1 = np.average(window1, axis=(1, 2, 3, 4))#np.average(window1[left_window_idx])
    #     avg2 = np.average(window2, axis=(1, 2, 3, 4))
        
    #     #have standard deviation and average of all windows
    #     # s = np.sum(np.subtract(window1[left_window_idx], avg1) * (window2.transpose() - avg2).transpose(), axis=(1, 2, 3, 4))
         
    #     s = np.add(window1, axis=(1, 2, 3, 4))
    #     # values.append(s / ((Config.WINDOW_SIZE) ** 2 * std1 * std2))
    
    
    
    


if __name__ == "__main__":
    left_img_path = args.left
    right_img_path = args.right
    disparity_scale = args.scale
    output_file = args.out
    
    stero_img = stereo_reconstruction(left_img_path, right_img_path)
    
    cv2.imwrite(output_file, stero_img)