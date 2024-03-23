import cv2
import numpy as np
 
#images = [cv2.imread('images/hi_hat.jpg'),cv2.imread('images/bongo.jpg'), cv2.imread('images/snare.jpeg'), cv2.imread('images/timpani.jpeg')]

def drum_image(images,frame):
    """This function will add 4 drum images to a webcam viewer in OpenCV. 
    images: The 4 drum images in a list [] format.
    frame: the camera frame image from OpenCV to add the images to."""
    
    #creates the location for the drums
    row_loc = [330,330,330,330]
    col_loc = [0,160,320,480]
    i = 0

    # Resizes the drums to fit into each location on the video
    for ii in images:
        new_width =150
        new_height=150
        drum_size = (new_width, new_height)
        img2 = cv2.resize(ii, drum_size, interpolation = cv2.INTER_AREA)
        
    # Create an ROI in the location that the image is wanted.
        rows,cols,channels = img2.shape
        roi = frame[row_loc[i]:row_loc[i]+rows, col_loc[i]:col_loc[i]+cols ]

    # Create a mask of of the image to remove the background color.
        img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 230, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
    # Now black-out the background of the image in the ROI
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask)

    # Take only the image (not the background) from the drum image.
        img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)

    # Put the image in the ROI and modify the main frame image
        dst = cv2.add(roi,img2_fg)
        frame[row_loc[i]:row_loc[i]+rows, col_loc[i]:col_loc[i]+cols] = dst
        i = i+1
        
    i = 0
