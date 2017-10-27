""" This module contains functions and structures for processing poker chip images """

### Import necessary packages
import cv2
import os
import copy
import math
import numpy as np
import display as dp
from matplotlib import pyplot as plt

### Constants
MAX_NORM_DIFF = 0.17
MIN_CHIP_AREA = 1000
MAX_CHIP_AREA = 8000

### Public Functions ###

def detectChips(image):
    """ Returns contours of the chips in an image """

    # List to store all valid chips
    chip_cnts = []

    # Pre-processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow("blurred image", blur); cv2.waitKey(0); cv2.destroyAllWindows()

    # Threshold with Otsu's method
    (_, thresh) = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow("Thresholded playing area", thresh); cv2.waitKey(0); cv2.destroyAllWindows()

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("Closing", closing); cv2.waitKey(0); cv2.destroyAllWindows()

    # Find contours and sort by size
    (_, cnts, hier) = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

    # Initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []

    # Catch cases where no contours were detected
    try:

        # Fill empty lists with sorted contour and sorted hierarchy. 
        for i in index_sort:
            cnts_sort.append(cnts[i])
            hier_sort.append(hier[0][i])  

        for i in range(len(cnts_sort)):
        
            # Get the radius according to the area
            area = cv2.contourArea(cnts_sort[i])
            r1 = math.sqrt(area/math.pi)

            # Get the radius according to the perimeter
            perimeter = cv2.arcLength(cnts_sort[i],True)   
            r2 = perimeter/(2*math.pi)

            # Normalise the difference
            diff = abs(r1-r2)
            norm_diff = diff/np.mean([r1, r2])

            #print('area = {}, norm diff = {}'.format(area, norm_diff))
            #img_disp = copy.deepcopy(image)
            #cv2.drawContours(img_disp, [cnts_sort[i]], 0, dp.CYAN, 3)
            #cv2.imshow("Detected Chips", img_disp); cv2.waitKey(0); cv2.destroyAllWindows()

            # Circles have similar radii due to area and perimeter.    
            # Chip contours should have no parents.
            if ((norm_diff < MAX_NORM_DIFF) and ((hier_sort[i][3] == -1))
                and (area > MIN_CHIP_AREA) and (area < MAX_CHIP_AREA)):
                
                chip_cnts.append(cnts_sort[i])

    # If there are no contours, do nothing
    except:
        pass

    return chip_cnts

def drawChips(image, all_chips):

    cv2.drawContours(image, all_chips, -1, dp.CYAN, 3)

    return image

### Test Functions

def videoTest():
    """ Run the chip detector module by itself """
    
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,9999)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,9999)

    while(True):

        # Get the next frame    
        flag, img = cap.read()
        
        # Get a list of all of the contours around cards
        chip_cnts = findChips(img)
        img_disp = copy.deepcopy(img)

        # Show the display image
        cv2.imshow("Detected Chips", img_disp)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def imageTest():

    # Get next image of playing area
    img = cv2.imread(os.path.join('game_images', 'both2.png'))
    img_disp = copy.deepcopy(img)

    # Find all of the chips and draw them on the temp image
    all_chips = detectChips(img)
    img_disp = drawChips(img_disp, all_chips)

    cv2.imshow("Detected Chips", img_disp); cv2.waitKey(0); cv2.destroyAllWindows()

### Chip Module Test Code ###
if __name__ == "__main__":
    imageTest()