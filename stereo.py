import cv2
import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import time
from numba import jit

parser = argparse.ArgumentParser()
parser.add_argument("--dir")
parser.add_argument("--left")
parser.add_argument("--right")
parser.add_argument("--scale")
parser.add_argument("--out")
args = parser.parse_args()

class Config:
    FOCAL_LENGTH = 3740 #pixels
    BASELINE = 160 #mm
    WINDOW_SIZE = 10 #experiment with different sizes, ask what is typical?
    OCCLUSION_COST = 100000
    OCCLISION_FILLING = True
    DELTA = 1
    

def stereo_reconstruction(left_img_path, right_img_path, disparity_scale):
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    assert(left_img.shape == right_img.shape)
    
    left_stereo_img = np.zeros((left_img.shape[0], left_img.shape[1]), dtype=np.uint8)
    right_stereo_img = np.zeros((right_img.shape[0], right_img.shape[1]), dtype=np.uint8)

    left_windows = np.lib.stride_tricks.sliding_window_view(left_img, (Config.WINDOW_SIZE, Config.WINDOW_SIZE, 3))
    right_windows = np.lib.stride_tricks.sliding_window_view(right_img, (Config.WINDOW_SIZE, Config.WINDOW_SIZE, 3))    
    
    for scanline_idx in range(left_windows.shape[0]):
        
        window1_flat = np.reshape(left_windows[scanline_idx], (left_windows[scanline_idx].shape[0], -1))
        window2_flat = np.reshape(right_windows[scanline_idx], (right_windows[scanline_idx].shape[0], -1))
        
        path_matrix = get_path_matrix(window1_flat, window2_flat, Config.OCCLUSION_COST)
        left_disparity, right_disparity = find_best_path(path_matrix, window1_flat.shape[0] - 1, disparity_scale)
        
        left_stereo_img[scanline_idx, :left_disparity.shape[0]] = left_disparity #np.add(indexes - np.arange(indexes.shape[0]), disparity_scale)#Config.FOCAL_LENGTH * Config.BASELINE) / 
        right_stereo_img[scanline_idx, :right_disparity.shape[0]] = right_disparity
        
        # indexes = ssd(window1_flat, window2_flat)
        
        # left_stereo_img[scanline_idx, :indexes.shape[0]] = (indexes - np.arange(indexes.shape[0])) * disparity_scale
        # right_stereo_img[scanline_idx, :indexes.shape[0]] = (indexes - np.arange(indexes.shape[0])) * disparity_scale
        
        if scanline_idx%100 == 0:
            print('Scanline:', scanline_idx)
            # cv2.imshow('stereo', left_stereo_img)
            # cv2.waitKey(0)
   
    return left_stereo_img, right_stereo_img
                

@jit(nopython=True)
def get_path_matrix(window1, window2, occulusion):
    '''
    Gets cost matrix and path matrix 
    
    window1 (left) shape == (m,1,n,n,3)
    window2 (right) shape == (m,1,n,n,3)
    
    '''
    
    #flatten each into (1262, WINDOW_SIZExWINDOW_SIZEx3) 

    ssd = np.zeros((window1.shape[0], window2.shape[0]))
    cost_matrix = np.copy(ssd)
    path_matrix = np.zeros(ssd.shape)

    for col in range(window1.shape[0]):
        cost_matrix[col, 0] = col * occulusion
        cost_matrix[0, col] = col * occulusion

    for left_window_idx in range(window1.shape[0]):
        for right_window_idx in range(window2.shape[0]):
            
            #compare each left to the right
            ssd_val = np.sum((window1[left_window_idx] - window2[right_window_idx])**2) #score
            ssd[left_window_idx, right_window_idx] = ssd_val
            
            values = np.array([ 
                            cost_matrix[left_window_idx-1,right_window_idx-1] + ssd_val, #diagonal
                            cost_matrix[left_window_idx-1,right_window_idx] + occulusion, #occluded from left
                            cost_matrix[left_window_idx,right_window_idx-1] + occulusion]) #occluded from right
            
            cost_matrix[left_window_idx, right_window_idx] = np.min(values)
            path_matrix[left_window_idx, right_window_idx] = np.argmin(values)

    return path_matrix

@jit(nopython=True)
def find_best_path(path_matrix, num_cols, disparity_scale):
    
    left_disparity = np.zeros(num_cols)
    right_disparity = np.zeros(num_cols)
    
    i, j, last_left_disp, last_right_disp = num_cols, num_cols, 0, 0
    
    while(i > 1 and j > 1):     
        if path_matrix[i,j] == 0:
            last_left_disp = abs(i-j) * disparity_scale
            last_right_disp = abs(j-i) * disparity_scale
            
            left_disparity[i] = last_left_disp # Disparity Image in Left Image coordinates
            right_disparity[j] = last_right_disp # Disparity Image in Right Image coordinates
            i = i-1
            j = j-1   
        elif path_matrix[i, j] == 1:
            left_disparity[i] = last_left_disp
            i = i-1
        elif path_matrix[i,j] == 2:
            right_disparity[j] = last_right_disp
            j = j-1
            
    return left_disparity, right_disparity


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

    return np.argmin(ssd, axis=1) #each index is left pixel, each index value is right pixel

@jit(nopython=True)
def bmp_evalution(results, ground_truth, delta):
    '''
    Middlebury BMP evaluation metric
    '''
    
    assert(results.shape == ground_truth.shape)
    
    values = np.zeros(results.shape)
    
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            if abs((ground_truth[i, j, 0] - results[i, j, 0])) > delta:
                values[i, j] = 1.0
    
    return np.mean(values)

def plot_values(data, save_dir, labels):
    '''
    Bar graph with evalution on 4 different images
    Each plot of bar graph has 3 values, left dis, right dis, and accuracy... so 12 graphs
    '''
    
    fig, axs = plt.subplots(2)
    fig.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.5)
    
    x = np.arange(len(labels))

    axs[0].set_title('Left and Right Stereo BMP Evaluations')
    axs[0].set_ylabel('Bad Matched Pixel (BMP)')
    axs[0].set_xlabel('Images Dataset')
    axs[0].set_xticks(x + 0.125)
    axs[0].set_xticklabels(labels)
    axs[0].bar(x + 0.00, data[:, 0], width = 0.25, label='left') #left disparity
    axs[0].bar(x + 0.25, data[:, 1], width = 0.25, label='right') #right disparity
    axs[0].legend()
    
    axs[1].set_title('Time to Calculate Stereo Reconstruction')
    axs[1].set_ylabel('Time (minutes)')
    axs[1].set_xlabel('Images Dataset')
    axs[1].bar(labels, data[:, 2], width = 0.25) #time

    plt.savefig(os.path.join(save_dir, 'plots.png'))
    

if __name__ == "__main__":
    dir = args.dir
    left_path = args.left
    right_path = args.right
    disparity_scale = int(args.scale)
    out_dir = args.out
    #add eval argument
    
    for dirpath, dirnames, filenames in os.walk(dir):
        
        # print(dirpath, dirnames)
        metrics = np.zeros((len(dirnames), 3))
        
        for sub_dir in range(len(dirnames)):  
            current_dir = os.path.join(dir, dirnames[sub_dir])
            
            print('------------------------------')
            print(current_dir) 
            
            left_img_path = os.path.join(current_dir, left_path)
            right_img_path = os.path.join(current_dir, right_path)  
            
            start = time.time()
            left_stereo_img, right_stereo_img = stereo_reconstruction(left_img_path, right_img_path, disparity_scale)
            end = time.time()
            
            elapsed_time = (end-start)/60.0    
            
            if not os.path.exists(os.path.join(current_dir, out_dir)):
                os.mkdir(os.path.join(current_dir, out_dir))
            
            cv2.imwrite(os.path.join(current_dir, out_dir, 'res1.png'), right_stereo_img)
            cv2.imwrite(os.path.join(current_dir, out_dir, 'res5.png'), left_stereo_img)
            
            right_stereo_img = cv2.imread(os.path.join(current_dir, out_dir, 'res1.png'))
            left_stereo_img = cv2.imread(os.path.join(current_dir, out_dir, 'res5.png'))
              
            disp_right = cv2.imread(os.path.join(current_dir, 'disp1.png'))
            disp_left = cv2.imread(os.path.join(current_dir, 'disp5.png'))

            left_bmp = bmp_evalution(left_stereo_img, disp_left, Config.DELTA)
            right_bmp = bmp_evalution(right_stereo_img, disp_left, Config.DELTA)
                        
            print('Left BMP:', left_bmp)
            print('Right BMP:', right_bmp)
            print("Elapsed time:", elapsed_time)
            
            metrics[sub_dir] = np.array([left_bmp, right_bmp, elapsed_time])

        plot_values(metrics, dirpath, dirnames) 
        break

    