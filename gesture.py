""" This module contains functions and structures for processing blackjack hand gestures """

### Import packages
import cv2
import copy
import math
import numpy as np

### Constants ###

### Functions ###

def detect(frame):

    font = cv2.FONT_HERSHEY_SIMPLEX

    result, frame_contour = color_find_hand(frame)

    if result == 1:
        cv2.putText(frame_contour, 'Hit!!', (40, 400), font, 3,
                    (0, 255, 0), 2, cv2.LINE_AA)
    elif result == 2:
        cv2.putText(frame_contour, 'Stand!!', (40, 400), font, 3,
                    (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame_contour, 'Sign not recognised', (40, 400),
                    font, 2,
                    (0, 0, 255), 2, cv2.LINE_AA)

    return frame_contour

def color_find_hand(frame):

    show_process = 0
    # --------------------- Pre-processing stage ---------------------
    # img1 = img_proc.maxRGB(frame)
    # img2 = img_proc.grayWorld(frame)
    # img3 = img_proc.myMinkowski(frame)
    # cv2.namedWindow("equ", cv2.WINDOW_NORMAL)
    # cv2.imshow("equ", np.hstack([frame,img1,img2,img3]))
    # cv2.resizeWindow('equ', 1200, 700)

    # maxRGB is found to improve the contrast really well, but it slows down
    #  the process and doesn't improve results
    # frame = img_proc.maxRGB(frame)

    # ------------- Convert to desired colour space ------------------
    im_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

    # ------------------ Adjust the illuminance ----------------------
    im_smooth = copy.deepcopy(im_ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    im_smooth[:,:,0] = clahe.apply(im_smooth[:,:,0])
    # just for display
    im_ycrcb_disp = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2BGR)
    im_smooth_disp = cv2.cvtColor(im_smooth, cv2.COLOR_YCR_CB2BGR)
    if show_process:
        cv2.namedWindow("Illuminance correction", cv2.WINDOW_NORMAL)
        cv2.imshow("Illuminance correction", np.hstack([im_ycrcb_disp, im_smooth_disp]))
        cv2.resizeWindow("Illuminance correction", 1200, 700)

    # ------------------ Skin color tresholding ----------------------
    skin_ycrcb_min = np.array((0, 133, 77))
    skin_ycrcb_max = np.array((255, 190, 127))
    # Create a binary mask
    skin_ycrcb = cv2.inRange(im_smooth, skin_ycrcb_min, skin_ycrcb_max)

    # perform opening to remove noise and closing to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.morphologyEx(skin_ycrcb, cv2.MORPH_OPEN, kernel,
                                iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel,
                                iterations=2)

    # blur the mask to help remove noise
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    if show_process:
        cv2.namedWindow("skin mask", cv2.WINDOW_NORMAL)
        cv2.imshow("skin mask", np.hstack([skin_ycrcb, skin_mask]))
        cv2.resizeWindow("skin mask", 700, 700)

    # ------------------ Find image contour --------------------------
    im2_, contours, hierarchy_ = cv2.findContours(skin_mask, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)

    # find the maximum contour
    max_area = 0
    max_cont = 0
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_cont = i

    # if there's contour found, only do the largest one
    if contours:
        biggest_contour = contours[max_cont]

        # draw contour
        drawn_frame = copy.deepcopy(frame)
        cv2.drawContours(drawn_frame, contours, max_cont, (255, 0, 0), 3)

        # -------------- Matching to determine signal ---------------------
        # if we cant find any fingers, result will be 0 and draw_frame will
        # not change
        result, drawn_frame = match_defects(biggest_contour, drawn_frame)

    else:
        result = 0
        drawn_frame = frame

    return [result, drawn_frame]

def matching_Hu(contour, tresh_hit=0.15, tresh_stand=0.2):

    # -------- save current contour as template -----------
    # np.save('hit', biggest_contour)

    # -------- compare contour to saved template ----------
    hit_gest = np.load('hit.npy')
    stand_gest = np.load('stand.npy')
    strategy = cv2.CONTOURS_MATCH_I3

    match_hit = cv2.matchShapes(contour, hit_gest, strategy, 0)
    match_stand = cv2.matchShapes(contour, stand_gest, strategy, 0)

    # print('Match with hit: ', match_hit)
    # print('Match with stand: ', match_stand)
    # print('---')

    # determine the result of hit or stand
    result = 0
    if match_hit < tresh_hit:
        result = 1
    elif match_stand < tresh_stand:
        result = 2

    return result

def match_defects(contour,frame):

    # Find the convex hull and the defects
    hull = cv2.convexHull(contour,returnPoints = False)
    defects = cv2.convexityDefects(contour,hull)

    # additional check for hit match
    hullFit = cv2.convexHull(contour, returnPoints=True)
    (x, y), (MA, ma), angle = cv2.fitEllipse(hullFit)
    ratio = MA / ma
    # print('ratio:', ratio)

    # find the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 2)
    palm_region = y + h - (h / 2)

    num_points = 0
    
    # Run through all the contours and find the fingers
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # draw the convex hull
        cv2.line(frame, start, end, [255, 120, 255], 2)

        if d > 10000:
            # reject any points that is in the lower region of the bounding
            # box. they are likely hand or palm.
            if end[0] < palm_region:
                pass
            elif start[0] < palm_region:
                pass
            else:
                num_points += 1
                cv2.circle(frame, start, 10, [0, 0, 255], -1)
                cv2.circle(frame, far, 10, [150, 150, 0], -1)
                cv2.line(frame, start, far, [0, 255, 255], 2)

    # Determine the sign
    if num_points == 1 and ratio < 0.4:
        result = 1
    elif num_points in [4, 5, 6]:
        result = 2
    else:
        result = 0

    return [result, frame]

def eucl_distance(val1, val2):

    dist = math.sqrt((val1[0] - val2[0]) ** 2 + (val1[1] - val2[1]) ** 2)

    return dist

def findAngle(start, end, far):

    l1 = eucl_distance(far, start)
    l2 = eucl_distance(far, end)
    dot = (start[0] - far[0]) * (end[0] - far[0]) \
          + (start[1] - far[1]) * (end[1] - far[1])

    angle = math.acos(dot / (l1 * l2))
    angle = math.degrees(angle)

    return angle