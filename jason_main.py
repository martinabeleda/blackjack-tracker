# import standard libraries
import cv2
import sys
import imutils
import time
from copy import deepcopy
from datetime import datetime

# import own libraries
import surface


count = 20 + 1
curr_time = float(datetime.now().strftime('%s.%f')[9:-5])
# initialise an empty playing surface
playing_surface = None

while True:

    # Get image (replace with webcam feed)
    # image = cv2.imread('today4.png')  # example of playing surface too small
    image = cv2.imread('new_surf5.png')  # example of playing surface that is good

    # Configure the raw original image
    orig_disp = deepcopy(imutils.resize(image, height=300))

    # Start with a fresh copy of the raw original image
    timer_disp = deepcopy(orig_disp)
    test_time = float(datetime.now().strftime('%s.%f')[9:-5])
    if count != 0 and abs(test_time - curr_time) > 1:
        curr_time = float(datetime.now().strftime('%s.%f')[9:-5])
        count -= 1
    timer_disp = surface.timer(timer_disp, count)

    if playing_surface:
        # Configure and display the contoured and transformed images if the playing surface was found
        cnt_disp = deepcopy(imutils.resize(playing_surface.img_cnt, height=300))
        trans_disp = deepcopy(imutils.resize(playing_surface.transform, height=300))
        surface.display(timer_disp, cnt_disp, trans_disp)
    else:
        # Try and obtain a valid playing surface object
        playing_surface = surface.detect(image)
        not_found = surface.not_found(deepcopy(orig_disp))
        # Display the countdown timer and the raw original until a valid surface is found
        surface.display(timer_disp, not_found)

    # handling closing of displays in main atm, could shift anywhere though
    key = cv2.waitKey(delay=1)

    # keep displaying images until user enters 'q'
    if key == ord('q') or key == ord('Q'):
        cv2.destroyAllWindows()
        break


# save transformed surface (can comment out - was using for testing and sending to Marty)
cv2.imwrite('Playing_Surface_Transformed.png', playing_surface.transform)

sys.exit(0)

