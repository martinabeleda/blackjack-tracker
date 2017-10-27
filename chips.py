""" This module contains functions and structures for processing poker chip images """

### Import necessary packages
import cv2
import os
import copy
import math
import numpy as np
import display as dp
from matplotlib import pyplot as plt

def findChips(img):
    """ Returns contours of the chips in an image """

    chip_cnts = []

    # Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow("blurred image", blur); cv2.waitKey(0); cv2.destroyAllWindows()

    # Threshold with Otsu's method
    (_, thresh) = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow("Thresholded playing area", thresh); cv2.waitKey(0); cv2.destroyAllWindows()

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closing", closing); cv2.waitKey(0); cv2.destroyAllWindows()

    # Morphological dilation
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    dilation = cv2.dilate(closing,kernel,iterations = 1)
    cv2.imshow("Closing", dilation); cv2.waitKey(0); cv2.destroyAllWindows()
    """

    # Find contours and sort by size
    (_, cnts, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

    # Catch cases where no contours were detected
    try:

        # Initialize empty sorted contour and hierarchy lists
        cnts_sort = []
        hier_sort = []

        # Fill empty lists with sorted contour and sorted hierarchy. 
        for i in index_sort:
            cnts_sort.append(cnts[i])
            hier_sort.append(hier[0][i])  

        for i in range(len(cnts)):
        
            area = cv2.contourArea(cnts[i])
            perimeter = cv2.arcLength(cnts[i],True)

            r1 = math.sqrt(area/math.pi)
            r2 = perimeter/(2*math.pi)
            diff = abs(r1-r2)
            norm_diff = diff/np.mean([r1, r2])    

            print('Diff = {}, Norm Diff = {}, Card {}'.format(abs(r1-r2), ))

            ### Drawing ##
            img_draw = copy.deepcopy(img)
            cv2.drawContours(img_draw, [cnts[i]], -1, dp.GREEN, 3)
            cv2.imshow("Contours", img_draw); cv2.waitKey(0); cv2.destroyAllWindows()

    # If there are no contours, do nothing
    except:
        pass

    return chip_cnts

def main():
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


def test():

    # Get next image of playing area
    img = cv2.imread(os.path.join('game_images', 'chips1.png'))

    # Find all of the chips
    chip_cnts = findChips(img)

### Chip Module Test Code ###
if __name__ == "__main__":
    test()