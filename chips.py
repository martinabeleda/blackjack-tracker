""" This module contains functions and structures for processing poker chip images """

### Import necessary packages
import cv2
import os
import copy
import math
import numpy as np
import display as dp

### Constants
MAX_NORM_DIFF = 0.2
MIN_CHIP_AREA = 1000
MAX_CHIP_AREA = 8000

### Structures ###
class chip:
    """Structure to store information about chips in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of chip
        self.center = [] # Center point of chip
        self.radius = [] # Radius of the chip
        self.norm_diff = [] # Normalised difference between area and perimeter calculated radii

### Public Functions ###

def detect(image):
    """ Returns contours of the chips in an image """

    # List to store all valid chip objects
    all_chips = []

    # Pre-processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow("blurred image", blur); cv2.waitKey(0); cv2.destroyAllWindows()

    # Threshold with Otsu's method
    (_, thresh) = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

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
            mean_diff = np.mean([r1, r2])
            if mean_diff == 0:
                norm_diff = MAX_NORM_DIFF + 1
            else:
                norm_diff = diff/mean_diff

            # Circles have similar radii due to area and perimeter.    
            # Chip contours should have no parents.
            if ((norm_diff < MAX_NORM_DIFF) and ((hier_sort[i][3] == -1))
                and (area > MIN_CHIP_AREA) and (area < MAX_CHIP_AREA)):

                new_chip = chip()

                new_chip.contour = cnts_sort[i]
                new_chip.norm_diff = norm_diff

                (x,y),radius = cv2.minEnclosingCircle(cnts_sort[i])
                new_chip.center = (int(x),int(y))
                new_chip.radius = int(radius)

                all_chips.append(new_chip)

    # If there are no contours, do nothing
    except:
        pass

    return all_chips

def display(image, all_chips):

    for i in range(len(all_chips)):
        
        cv2.circle(image, all_chips[i].center, all_chips[i].radius, dp.CYAN, 2)

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
        img_disp = copy.deepcopy(img)

        # Find all of the chips and draw them on the temp image
        all_chips = detect(img)
        img_disp = display(img_disp, all_chips)
        
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
    all_chips = detect(img)
    img_disp = display(img_disp, all_chips)

    cv2.imshow("Detected Chips", img_disp); cv2.waitKey(0); cv2.destroyAllWindows()

### Chip Module Test Code ###
if __name__ == "__main__":
    imageTest()