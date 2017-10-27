# import standard libraries
import cv2
import sys
import imutils
from copy import deepcopy

# import own libraries
import surface


while True:

    # get image (replace with webcam feed)
    image = cv2.imread('new_surf5.png')

    # obtain playing surface object
    playing_surface = surface.detect(image)

    # configure images for display and then display them
    orig_disp = deepcopy(imutils.resize(image, height=300))
    cnt_disp = deepcopy(imutils.resize(playing_surface.img_cnt, height=300))
    trans_disp = deepcopy(imutils.resize(playing_surface.transform, height=300))
    surface.display(orig_disp, cnt_disp, trans_disp)

    # handling closing of displays in main atm, could shift anywhere though
    key = cv2.waitKey(delay=1)

    # keep displaying images until user enters 'q'
    if key == ord('q') or key == ord('Q'):
        cv2.destroyAllWindows()
        break

# save transformed surface (can comment out - was using for testing and sending to Marty)
cv2.imwrite('Playing_Surface_Transformed.png', playing_surface.transform)

sys.exit(0)

