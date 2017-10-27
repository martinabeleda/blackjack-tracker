""" This program takes an image of a blackjack playing surface and identifies the cards and chips """

### Import necessary packages
import os
import cv2
import copy
import cards as cd
import chips as ch
import display as dp

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
        all_cards = cd.detectCards(img, rank_path)
        img_disp = cd.drawCards(img_disp, all_cards)

        # Find all of the chips and draw them on the temp image
        all_chips = ch.detectChips(img)
        img_disp = ch.drawChips(img_disp, all_chips)

        cv2.imshow("Detected Cards and Chips", img_disp); cv2.waitKey(0); cv2.destroyAllWindows()
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def imageTest():

    # Get next image of playing area
    img = cv2.imread(os.path.join('game_images', 'both2.png'))
    img_disp = copy.deepcopy(img)

    # Get a list of card objects in the image and draw on temp image
    all_cards = cd.detectCards(img, rank_path)
    img_disp = cd.drawCards(img_disp, all_cards)

    # Find all of the chips and draw them on the temp image
    all_chips = ch.detectChips(img)
    img_disp = ch.drawChips(img_disp, all_chips)

    cv2.imshow("Detected Cards and Chips", img_disp); cv2.waitKey(0); cv2.destroyAllWindows()

### Module Test Code ###
if __name__ == "__main__":
    imageTest()