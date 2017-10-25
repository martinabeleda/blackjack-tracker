""" This program takes an image of a blackjack playing surface and identifies the cards """

### Import necessary packages
import os
import cv2
import copy
import cards

### Constants
rank_path = "card_images"
font = cv2.FONT_HERSHEY_SIMPLEX

### Structures

### Main code body

# Load the card rank images into a list of rank objects
ranks = cards.load_ranks(rank_path)

# Get next image of playing area
img = cv2.imread(os.path.join('game_images', 'transformed_small1.png'))

# Get a list of all of the contours around cards
all_cards = cards.findCards(img)

for i in range(len(all_cards)):

    # Produce a top-down image of each card
    all_cards[i].processCard(img)

    # Find the best rank match for this card
    all_cards[i].matchRank(ranks, cards.TEMPLATE_MATCHING)

    ### Display ###
    img_disp = copy.deepcopy(img)
    cv2.drawContours(img_disp, [all_cards[i].contour], 0, (0,255,0), 2)
    text_pos = (all_cards[i].center[0], all_cards[i].center[1])
    cv2.putText(img_disp, all_cards[i].best_rank_match, text_pos, font, 0.5, (255,0,0), 1, cv2.LINE_AA)
    cv2.imshow("Detected Cards", img_disp); cv2.waitKey(0); cv2.destroyAllWindows()
