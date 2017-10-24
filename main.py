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
img = cv2.imread(os.path.join('game_images', 'transformed.png'))

# Get a list of all of the contours around cards
all_cards = cards.findCards(img)

for i in range(len(all_cards)):

    # Produce a top-down image of each card
    all_cards[i] = cards.processCard(all_cards[i], img)

    # Find the best rank match for this card
    all_cards[i].matchRank(ranks)

    print(all_cards[i].best_rank_match)

# Create a copy of the image for display
img_disp = copy.deepcopy(img)
cv2.drawContours(rank_col, cnts, 0, (0,255,0), 2)
cv2.imshow("Detected Cards", img_disp); cv2.waitKey(0); cv2.destroyAllWindows()

