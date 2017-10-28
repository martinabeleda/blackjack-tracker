# import standard libraries
import cv2
import sys
import imutils
import time
from copy import deepcopy
from datetime import datetime

# import own libraries
import surface


count = 10 + 1
curr_time = float(datetime.now().strftime('%s.%f')[9:-5])
# initialise an empty playing surface
valid_surface = None

while count != 0:

    # Initialise or reinitialise the surface each time we grab a new frame
    playing_surface = None

    # Get image (replace with webcam feed)
    if count % 2 == 0:
        image = cv2.imread('/Users/jasonwebb/PycharmProjects/Vision_Major/transformed3_uhoh.png')  # example of playing surface too small
    else:
        image = cv2.imread('/Users/jasonwebb/PycharmProjects/Vision_Major/transformed3_uhoh.png')  # example of playing surface that is good

    # Try and obtain a valid playing surface object
    playing_surface = surface.detect(image)

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
        valid_surface = playing_surface.transform
    else:
        if valid_surface:
            cv2.destroyWindow("Transformed")
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
if valid_surface:
    cv2.imwrite('Last_Valid_Surface.png', valid_surface)
cv2.destroyAllWindows()
sys.exit(0)

