""" This program takes an image of a blackjack playing surface and identifies the cards and chips """

### Import necessary packages
import os
import cv2
import copy
import imutils
import display as dp

# import own libraries
import surface
import cards
import chips

### Constants

## import jason libraries
import initialise

## import alvin libraries
from mainColorTresh import findGesture

rank_path = "card_images"
font = cv2.FONT_HERSHEY_SIMPLEX


def videoTest():
    """ Run the chip detector module by itself """

    # set up the camera and set max resolution
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)

    # Run through countdown and grab playing surface. Returns an surface
    # object.
    surface = initialise.get_surface(cap, 15)

    state = 0
    # If initialisation found a successful transform, else exit the program
    if surface is not None:
        while (True):

            # Get the next frame
            (flag, img) = cap.read()
            # Transform using the transformation matrix found during
            # initialisation
            transformed = cv2.warpPerspective(img, surface.perspective_matrix,
                                              (surface.width, surface.height))

            if state == 0:
                img_disp = copy.deepcopy(transformed)

                # Get a list of card objects in the image and draw on temp image
                all_cards = cards.detect(transformed, rank_path)
                img_disp = cards.display(img_disp, all_cards)

                # Find all of the chips and draw them on the temp image
                all_chips = chips.detect(transformed)
                img_disp = chips.display(img_disp, all_chips)

                # configure images for display and then display them
                cnt_disp = copy.deepcopy(imutils.resize(surface.img_cnt,
                                                        height=400))

                cv2.imshow("Playing surface contour", cnt_disp)
                cv2.imshow("Detected Cards and Chips", img_disp)

                key = cv2.waitKey(delay=1)

                if key == ord('t'):
                    cv2.destroyAllWindows()
                    state = not state
                elif key == ord('q'):
                    break

            elif state == 1:
                frame_contour = findGesture(transformed)

                # Show final image with contour
                # cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)
                cv2.imshow("Gesture Recognition", frame_contour)
                cv2.resizeWindow("Gesture Recognition", 700, 700)

                key = cv2.waitKey(delay=1)

                if key == ord('t'):
                    cv2.destroyAllWindows()
                    state = not state
                elif key == ord('q'):
                    break

    else:
        print('Initialisation failed. No successful transform found')
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def imageTest():

    # Get next image of playing area
    img = cv2.imread(os.path.join('game_images', 'surface2.png'))

    # obtain playing surface object
    playing_surface = surface.detect(img)
    transformed = playing_surface.transform
    img_disp = copy.deepcopy(transformed)

    # Get a list of card objects in the image and draw on temp image
    all_cards = cards.detect(transformed, rank_path)
    img_disp = cards.display(img_disp, all_cards)

    # Find all of the chips and draw them on the temp image
    all_chips = chips.detect(transformed)
    img_disp = chips.display(img_disp, all_chips)

    # configure images for display and then display them
    cnt_disp = copy.deepcopy(imutils.resize(playing_surface.img_cnt, height=300))

    cv2.imshow("Playing surface contour", cnt_disp)
    cv2.imshow("Detected Cards and Chips", img_disp); cv2.waitKey(0); cv2.destroyAllWindows()

### Module Test Code ###
if __name__ == "__main__":
    imageTest()
