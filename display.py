import cv2
from copy import deepcopy

""" This module containts constants and functions for display """

# Drawing
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
CYAN = (255,255,0)
MAGENTA = (255,0,255)


def regions(image, surface_obj, bg=True):
    # Add the semi-transparent bg boxes if requested
    if bg:
        overlay = deepcopy(image)
        cv2.rectangle(overlay, (0, 0), (220, 80), (0, 0, 0), -1)
        cv2.rectangle(overlay, (surface_obj.player_region[0], 0), (surface_obj.player_region[0] + 220, 80), (0, 0, 0), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, image, 1 - alpha,
                        0, image)
    # Setup offsets for value printing
    x_offset = 10
    y_offset = 60
    shadow_offset = 2
    # Draw a line down the middle of the surface
    cv2.line(image, (surface_obj.dealer_region[1] + shadow_offset, 0),
             (surface_obj.dealer_region[1], int(surface_obj.height)), (0, 0, 0), 2)
    cv2.line(image, (surface_obj.dealer_region[1], 0),
             (surface_obj.dealer_region[1], int(surface_obj.height)), (255, 255, 255), 2)
    # Add titles to each region of the display
    cv2.putText(image, 'Dealer', (x_offset + shadow_offset, y_offset + shadow_offset), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, 'Dealer', (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(image, 'Player',
                (surface_obj.dealer_region[1] + x_offset + shadow_offset, y_offset + shadow_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, 'Player', (surface_obj.dealer_region[1] + x_offset, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    return


def hand_values(image, surface_obj, cards, bg=True, padded=True):
    dealer_tally = 0
    dealer_aces = 0
    player_tally = 0
    player_aces = 0

    # Testing area for getting value of each hand
    # Loop through each found card
    for i in range(len(cards)):
        # Set a flag if the current card is an ace
        if cards[i].best_rank_match == 'Ace':
            ace_flag = 1
        else:
            ace_flag = 0
        # Check if the card belongs to the dealer or the player (based on x centroid) and assign as appropriate
        if cards[i].center[0] >= surface_obj.dealer_region[0] and cards[i].center[0] <= \
                surface_obj.dealer_region[1]:
            dealer_tally += cards[i].value
            if ace_flag:
                dealer_aces += 1
        elif cards[i].center[0] >= surface_obj.player_region[0] and cards[i].center[0] <= \
                surface_obj.player_region[1]:
            player_tally += cards[i].value
            if ace_flag:
                player_aces += 1

    value_1 = dealer_tally
    value_2 = player_tally

    # Pad the values with a leading zero if optional padded argument is set
    # Convert to string in either case
    if padded:
        val_disp_1 = str(format(value_1, '02d'))
        val_disp_2 = str(format(value_2, '02d'))
    else:
        val_disp_1 = str(value_1)
        val_disp_2 = str(value_2)
    # Add the semi-transparent bg boxes if requested
    if bg:
        overlay = deepcopy(image)
        cv2.rectangle(overlay, (surface_obj.dealer_region[1] - 120, 0), (surface_obj.dealer_region[1], 80), (0, 0, 0), -1)
        cv2.rectangle(overlay, (surface_obj.player_region[1] - 120 , 0), (surface_obj.player_region[1], 80), (0, 0, 0), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, image, 1 - alpha,
                        0, image)
    # Setup offsets for value printing
    x_offset = surface_obj.dealer_region[1] - 100
    y_offset = 60
    shadow_offset = 2
    # Add the hand values to the display
    cv2.putText(image, val_disp_1, (x_offset + shadow_offset, y_offset + shadow_offset), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, val_disp_1, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(image, val_disp_2, (surface_obj.dealer_region[1] + x_offset + shadow_offset, y_offset + shadow_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, val_disp_2, (surface_obj.dealer_region[1] + x_offset, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    return
