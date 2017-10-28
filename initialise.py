# import standard libraries
import cv2
import sys
import imutils
import time
from copy import deepcopy
from datetime import datetime

# import own libraries
import surface


def get_surface(cap, count):
    count += 1
    curr_time = float(datetime.now().strftime('%s.%f')[9:-5])
    # initialise an empty playing surface
    valid_surface = None

    while count != 0:

        # Initialise or reinitialise the surface each time we grab a new frame
        playing_surface = None

        # take camera image
        (flag_, image) = cap.read()

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
            valid_surface = playing_surface
        else:
            # Add a text overlay in place of the contour image
            not_found = surface.not_found(deepcopy(orig_disp))
            if valid_surface is not None:
                surface.display(timer_disp, not_found, imutils.resize(valid_surface.transform, height=300))
            else:
                # Display the countdown timer and the raw original until a valid surface is found
                surface.display(timer_disp, not_found)

        # handling closing of displays in main atm, could shift anywhere though
        key = cv2.waitKey(delay=1)

        # keep displaying images until user enters 'q'
        if key == ord('a') or key == ord('A'):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()

    return valid_surface

