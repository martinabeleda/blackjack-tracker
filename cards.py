""" This module contains functions and structures for processing playing card images """

### Import packages
import cv2
import os
import copy
import imutils
import numpy as np
import display as dp

### Constants ###

# Debugging
ALVIN_LOVES_DEBUG = 0
WRITE_IMAGES = 0

# Card dimensions
CARD_MAX_AREA = 50000
CARD_MIN_AREA = 2000

CARD_MAX_PERIM = 1000
CARD_MIN_PERIM = 200

CORNER_HEIGHT = 80
CORNER_WIDTH = 50

RANK_HEIGHT = 125
RANK_WIDTH = 70

CARD_WIDTH = 200
CARD_HEIGHT = 300

# Polymetric approximation accuracy scaling factor
POLY_ACC_CONST = 0.02

# Matching algorithms
HU_MOMENTS = 0
TEMPLATE_MATCHING = 1
MAX_MATCH_SCORE = 2200

### Structures ###

class rank:
    """Structure to store information about each card rank."""

    def __init__(self):
        self.name = "rank_name"
        self.img = []  # Thresholded image of card rank
        self.contour = []  # Contour of rank
        self.value = 0  # Value of rank

class card:
    """Structure to store information about cards in the camera image."""

    def __init__(self):
        self.contour = []  # Contour of card
        self.corner_pts = []  # Corner points of card
        self.center = []  # Center point of card
        self.img = []  # 200x300, flattened, grayed, blurred image
        self.rank_img = []  # Thresholded, sized image of card's rank
        self.rank_contour = []  # Contour of the rank
        self.best_rank_match = "Unknown"  # Best matched rank
        self.rank_score = 0  # Difference between rank image and best matched train rank image
        self.value = 0  # Same as best rank, but in numerical form

    def processCard(self, image):
        """ This function takes an image and contour associated with a card and returns a top-down image of the card """

        # Find width and height of card's bounding rectangle
        x, y, w, h = cv2.boundingRect(self.contour)
        self.width, self.height = w, h

        # Find the centre of the card
        pts = self.corner_pts
        average = np.sum(pts, axis=0)/len(pts)
        cent_x = int(average[0][0])
        cent_y = int(average[0][1])
        self.center = [cent_x, cent_y]   

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	

        # Create a flattened image of the isolated card
        self.img = flattener(gray, pts, w, h)
        if ALVIN_LOVES_DEBUG:
            cv2.imshow("This card flattened", self.img); cv2.waitKey(0); cv2.destroyAllWindows()

        # Crop the corner from the card start from (5,5) to cut out dark edges
        rank_img = self.img[5:CORNER_HEIGHT, 5:CORNER_WIDTH]
        if ALVIN_LOVES_DEBUG:
            cv2.imshow("This rank", rank_img); cv2.waitKey(0); cv2.destroyAllWindows()

        # Assuming that a majority of the area is the background colour of the card, pad
        # out the edges with the median intensity value
        pad_value = np.median(rank_img)
        rank_img_padded = np.pad(rank_img, 5, 'constant', constant_values=pad_value)
        if ALVIN_LOVES_DEBUG:
            cv2.imshow("This rank padded", rank_img_padded); cv2.waitKey(0); cv2.destroyAllWindows()

        # Thresholding using Otsu's method
        (_, thresh) = cv2.threshold(rank_img_padded, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)
        if ALVIN_LOVES_DEBUG:
            cv2.imshow("This rank thresh", thresh); cv2.waitKey(0); cv2.destroyAllWindows()

        # Find the largest contour
        temp_thresh = copy.deepcopy(thresh)
        (_, this_rank_cnts, _) = cv2.findContours(temp_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        this_rank_cnts = sorted(this_rank_cnts, key=cv2.contourArea,reverse=True)

        # Get the bounding box around the rank, and resize to the template size
        if len(this_rank_cnts) != 0:
            
            self.rank_contour = this_rank_cnts[0]
            x1,y1,w1,h1 = cv2.boundingRect(this_rank_cnts[0])
            rank_crop = thresh[y1:y1+h1, x1:x1+w1]
            self.rank_img = cv2.resize(rank_crop, (RANK_WIDTH,RANK_HEIGHT), 0, 0)

            if ALVIN_LOVES_DEBUG:
                cv2.imshow("Cropped Rank", self.rank_img); cv2.waitKey(0); cv2.destroyAllWindows()
            
            if WRITE_IMAGES:
                cv2.imwrite('img.png', self.rank_img)

    def matchRank(self, all_ranks, match_method, last_cards):
        """ This function returns the best rank match of a given card image """
        
        # List to store rank match scores
        match_scores = [];

        for i in range(len(all_ranks)):
            if match_method is HU_MOMENTS:
                # Compare contours of the card with the template       
                match_scores.append(cv2.matchShapes(self.contour, all_ranks[i].contour, 1, 0.0))
            
            elif match_method is TEMPLATE_MATCHING:
                # Difference the card with the template
                diff_img = cv2.absdiff(self.rank_img, all_ranks[i].img)
                match_scores.append(int(np.sum(diff_img)/255))    
    
        ind = np.argmin(match_scores)      
        self.rank_score = match_scores[ind]

        # Determine if this is a valid match    
        if self.rank_score < MAX_MATCH_SCORE:
            self.best_rank_match = all_ranks[ind].name
            self.value = all_ranks[ind].value

        # Look at the previous cards list to help reduce flickering
        if ((self.best_rank_match == "Unknown") and (len(last_cards) != 0)):

            # Find a card in the last image that matches the coordinate of this card
            for i in range(len(last_cards)):

                cent_x, cent_y = self.center
                last_x, last_y = last_cards[i].center

                if ((abs(cent_x-last_x) < 10) and (abs(cent_y-last_y)<10)):
                    self.best_rank_match = last_cards[i].best_rank_match
                    self.value = last_cards[i].value

### Public Functions ###

def detect(image, rank_path, last_cards):
    """ Returns a list of card objects containing the cards within a given image """
    
    # Load the card rank images into a list of rank objects
    ranks = loadRanks(rank_path)

    # Get a list of all of the contours around cards
    all_cards = findCards(image)

    for i in range(len(all_cards)):

        # Produce a top-down image of each card
        all_cards[i].processCard(image)

        # Find the best rank match for this card
        all_cards[i].matchRank(ranks, TEMPLATE_MATCHING, last_cards)

    return all_cards

def display(image, all_cards):
    """ Draw cards from card objects onto playing area image """

    for i in range(len(all_cards)):

        # Draw on the temporary image
        if all_cards[i].best_rank_match == "Unknown":
            cnt_col = dp.RED
        else:
            cnt_col = dp.GREEN
        
        cv2.drawContours(image, [all_cards[i].contour], 0, cnt_col, 2)
        text_pos = (all_cards[i].center[0]-20, all_cards[i].center[1])
        cv2.putText(image, all_cards[i].best_rank_match, (text_pos[0] + 2,
                                                          text_pos[1] + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(image, all_cards[i].best_rank_match, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, dp.MAGENTA, 2, cv2.LINE_AA)

    return image

### Private Functions ###

def findCards(image):
    """ This function takes an images and returns a list of card objects with contour and corner info """

    # List to store card objects
    card_info = []

    # Convert to grayscale0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	

    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Threshold with Otsu's method
    (_, thresh) = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Find contours and sort by size
    (_, cnts, hier) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

    # Catch cases where no contours were detected
    try:

        # Initialize empty sorted contour and hierarchy lists
        cnts_sort = []
        hier_sort = []

        # Fill empty lists with sorted contour and sorted hierarchy. 
        for i in index_sort:
            cnts_sort.append(cnts[i])
            hier_sort.append(hier[0][i])  

        # Determine which of the contours are cards    
        for i in range(len(cnts_sort)):

            # Get the size of the cards
            size = cv2.contourArea(cnts_sort[i])

            # Use the perimeter of the card to set the accuracy parameter of the polymetric approximation
            peri = cv2.arcLength(cnts_sort[i],True)
            accuracy = POLY_ACC_CONST*peri    

            # Approximate the shape of the contours            
            approx = cv2.approxPolyDP(cnts_sort[i], accuracy, True)

            # Cards are determined to have an area within a given range,
            # have 4 corners and have no parents
            if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA) 
                and (len(approx) == 4) and (hier_sort[i][3] == -1)
                and (peri < CARD_MAX_PERIM)):
                new_card = card()
                new_card.contour = cnts_sort[i]  
                new_card.corner_pts = np.float32(approx)

                # Add the new card to the list
                card_info.append(new_card)

    # If there are no contours, do nothing
    except:
        pass

    return card_info

def loadRanks(path):
    """ Load rank images from a specified path. Store rank images in a list of rank objects """

    ranks = []
    rank_names = ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']
    rank_values = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    for name in rank_names:

        # Create a new instance of the rank class
        new_rank = rank()

        # Read the image of the rank
        img_path = os.path.join(path, name+'.png')
        new_rank.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Store the name
        new_rank.name = name

        # Get index of the name and store the associated value
        index = [i for i, x in enumerate(rank_names) if x == name]
        new_rank.value = rank_values[index[0]]

        # Store the largest contour
        temp = copy.deepcopy(new_rank.img)
        (_, cnts, _) = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea,reverse=True)
        new_rank.contour = cnts[0]

        # Add to the list
        ranks.append(new_rank)

    return ranks

def flattener(image, pts, w, h):
    """ Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image."""

    # rect to contain an array [top left, top right, bottom right, bottom left]
    rect = np.zeros((4,2), dtype = "float32")    
    s = np.sum(pts, axis = 2)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    temp = None
    new_pts = None
    for i in range(pts.shape[0]):
        if np.array_equal(pts[i], [rect[0]]):
            temp = np.delete(pts, i, 0)
            break

    for i in range(temp.shape[0]):
        if np.array_equal(temp[i], [rect[2]]):
            new_pts = np.delete(temp, i, 0)
            break

    # compute the difference between the points -- the top-right will have the minimum difference
    # and the bottom-left will have the maximum difference
    y_pts = new_pts[:,:,1]
    if y_pts[0] > y_pts[1]:
        rect[3] = new_pts[0]
        rect[1] = new_pts[1]
    else:
        rect[3] = new_pts[1]
        rect[1] = new_pts[0]

    # Extract the top left, top right, bottom right and bottom left
    (tl, tr, br, bl) = rect

    # Compute the width of the new image 
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to map the screen to a top-down,
    # "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # calculate the perspective transform matrix and warp the perspective to grab the target
    M = cv2.getPerspectiveTransform(rect, dst)
    playing_card = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    h = playing_card.shape[0]
    w = playing_card.shape[1]
    crop_amt = 0.25

    tl_crop = playing_card[0:int(h * crop_amt), 0:int(w * crop_amt)]
    tr_crop = playing_card[0:int(h * crop_amt), int(w * (1 - crop_amt)):w]

    if np.sum(tl_crop) > np.sum(tr_crop):

        playing_card = imutils.rotate_bound(playing_card, 90)

    resized = cv2.resize(playing_card, (CARD_WIDTH, CARD_HEIGHT))

    return resized

### Test Functions ###

def videoTest():
    """ Run the card detector module by itself """
    
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,9999)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,9999)

    # Load the card rank images into a list of rank objects
    rank_path = "rank_images"
    ranks = loadRanks(rank_path)

    while(True):

        # Get the next frame    
        flag, img = cap.read()
        
        # Get a list of all of the contours around cards
        all_cards = findCards(img)
        img_disp = copy.deepcopy(img)

        # Get a list of card objects in the image and draw on temp image
        all_cards = detect(img, rank_path)
        img_disp = display(img_disp, all_cards)

        # Show the display image
        cv2.imshow("Detected Cards", img_disp)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def imageTest():
    """ Test the cards module on a single flattened image """

    rank_path = "rank_images"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get next image of playing area
    img = cv2.imread(os.path.join('game_images', 'surface4.png'))
    img_disp = copy.deepcopy(img)

    # Get a list of card objects in the image and draw on temp image
    all_cards = detect(img, rank_path)
    img_disp = display(img_disp, all_cards)

    # Show the display image
    cv2.imshow("Detected Cards", img_disp); cv2.waitKey(0); cv2.destroyAllWindows()

### Cards Module Test Code ###
if __name__ == "__main__":
    imageTest()