# Algorithms

I decided to implement the Maximum Likelihood Stereo Algorithm [[1]](#1) to enhance the quality of my disparity maps compared to a baseline SSD. This algorithm takes a Disparity Space Image (DSI) of a scanline between two stereo images and uses dynamic programming to get the shortest path (lowest sum of dissimilarity scores) from the first to the last column. Once this shortest path is calculated, we can choose to include occlusion filling to further increase the quality of the disparity maps.

# Running Program

In order to run this program, simply execute the command: 
```
python3 stereo.py --dir [head directory of images] 
                  --left [left image] 
                  --right [right image] 
                  --out [output directory] 
                  --scale [disparity scale]
```
For this command, the ```--dir``` flag should be the head directory of images that contains subdirectories 
 with the downloaded/test datasets from Middlebury [[2]](#2). The program goes through all the subdirectories in ```--dir```, gets right and left disparity images, saves left and right disparity  maps into the out folder in the subdirectory, and generates a bar graph of the Bad Matched Pixels (BMP) and the runtime to calculate the maps for each of the different datasets. For the full-size images (1282x1110), the ```--scale``` should be 1; for half-size, the ```--scale``` should be 2; for third-size, it should be 3.

 A sample command is: 
 ```
 python3 stereo.py --dir ./half_images --left view5.png --right view1.png --out out --scale 2
 ```

## References
<a id="1">[1]</a> Cox, I.J., Hingorani, S.L., Rao, S., & Maggs, B.M. (1996). A Maximum Likelihood Stereo Algorithm. Comput. Vis. Image Underst., 63, 542-567.

<a id="2">[2]</a> Hirschmüller, H., & Scharstein, D. (2007). Evaluation of Cost Functions for Stereo Matching. 2007 IEEE Conference on Computer Vision and Pattern Recognition, 1-8.