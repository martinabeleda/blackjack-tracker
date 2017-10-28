""" This program takes an image of a blackjack playing surface and identifies the cards and chips """

### Import necessary packages
import os
import cv2
import copy
import display as dp

# import own libraries
import surface
import cards
import chips

### Constants

rank_path = "card_images"
font = cv2.FONT_HERSHEY_SIMPLEX

def videoTest():
    """ Run the chip detector module by itself """
    
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,9999)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,9999)

    while(True):

        # Get the next frame    
        (flag, img) = cap.read()
        img_disp = copy.deepcopy(img)

        # Get a list of card objects in the image and draw on temp image
        all_cards = cards.detect(img, rank_path)
        img_disp = cards.display(img_disp, all_cards)

        # Find all of the chips and draw them on the temp image
        all_chips = chips.detect(img)
        img_disp = chips.display(img_disp, all_chips)

        cv2.imshow("Detected Cards and Chips", img_disp); cv2.waitKey(0); cv2.destroyAllWindows()
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def imageTest():

    # Get next image of playing area
    img = cv2.imread(os.path.join('game_images', 'surface4.png'))

    # obtain playing surface object
    playing_surface = surface.detect(img)
    img_disp = copy.deepcopy(playing_surface.transform)

    # Get a list of card objects in the image and draw on temp image
    all_cards = cards.detect(playing_surface.transform, rank_path)
    img_disp = cards.display(img_disp, all_cards)

    # Find all of the chips and draw them on the temp image
    all_chips = chips.detect(img)
    img_disp = chips.display(img_disp, all_chips)

    # configure images for display and then display them
    orig_disp = deepcopy(imutils.resize(image, height=300))
    cnt_disp = deepcopy(imutils.resize(playing_surface.img_cnt, height=300))
    trans_disp = deepcopy(imutils.resize(playing_surface.transform, height=300))
    surface.display(orig_disp, cnt_disp, trans_disp)

    cv2.imshow("Detected Cards and Chips", img_disp); cv2.waitKey(0); cv2.destroyAllWindows()

### Module Test Code ###
if __name__ == "__main__":
    imageTest()