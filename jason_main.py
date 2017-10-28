# import standard libraries
import sys
import time
import cv2

# import own libraries
import initialise


# Run through countdown and grab playing surface
surface = initialise.get_surface(5)

time.sleep(3)

while True:

    # Apply the perspective transformation on the same image again and display them
    image = cv2.imread('new_surf5.png')

    cv2.imshow("Original", image)

    transformed = cv2.warpPerspective(image, surface.perspective_matrix, (surface.width, surface.height))

    cv2.imshow("Transform", transformed)
    # handling closing of displays in main atm, could shift anywhere though
    key = cv2.waitKey(delay=1)

    # keep displaying images until user enters 'q'
    if key == ord('q') or key == ord('Q'):
        cv2.destroyAllWindows()
        break


sys.exit(0)

