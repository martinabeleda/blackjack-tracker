""" This program takes an image of a blackjack playing surface and identifies the cards """

### Import necessary packages
import os
import cv2
import cards

### Constants
rank_path = "card_images"

### Structures

### Main code body

# Load the card rank images into a list of rank objects
ranks = cards.load_ranks(rank_path)

# Get next image of playing area
img = cv2.imread(os.path.join('game_images', 'transformed1.png'))

# Get a list of all of the contours around cards
all_cards = cards.get_card_contours(img)

for i in range(len(all_cards)):

    # Produce a top-down image of each card
    all_cards[i] = cards.process_card(all_cards[i], img)

    # Use template matching to get the rank of the card

    """
    cv2.imshow("This card image", all_cards[i].img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """ 