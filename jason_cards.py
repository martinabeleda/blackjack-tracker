def find_cards(image):

    card_count = 0

    min_contours = 0

    ratio = image.shape[0] / 300.0
    orig = image.copy()
    image = imutils.resize(image, height=300)

    # convert the image to grayscale, blur it, and find edges in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    cv2.imshow("gray", gray)
    cv2.imshow("edged", edged)
    # find contours in the edged image, keep only the largest ten, and initialize our screen contour
    _, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    card_contour = None

    while True:

        valid_countours = 0

        # loop over the contours
        for c in contours:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            curr_cnt = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then we can assume that we have found our target
            # this is because the contours are ordered largest to smallest already and we know we want the
            # largest rectangular contour
            if len(curr_cnt) == 4:
                valid_countours += 1
                if valid_countours > min_contours + card_count:
                    card_contour = curr_cnt
                    print(cv2.contourArea(curr_cnt))
                    break

        # now that we have our contour, we need to determine the top-left, top-right, bottom-right, and
        # bottom-left points so that we can later warp the image -- we'll start by reshaping our contour
        # to be our finals and initializing our output rectangle in top-left, top-right, bottom-right,
        # and bottom-left order
        pts = card_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        print(pts)
        # the top-left point has the smallest sum whereas the bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        for i in range(pts.shape[0]):
            if np.array_equal(pts[i], rect[0]):
                temp = np.delete(pts, i, 0)
                break

        for i in range(temp.shape[0]):
            if np.array_equal(temp[i], rect[2]):
                new_pts = np.delete(temp, i, 0)
                break

        # new_pts = pts
        # compute the difference between the points -- the top-right will have the minimum difference
        # and the bottom-left will have the maximum difference
        diff = np.diff(new_pts, axis=1)
        rect[1] = new_pts[np.argmin(diff)]
        rect[3] = new_pts[np.argmax(diff)]
        print(rect)
        # multiply the rectangle by the original ratio to get it back to the original size
        rect *= ratio

        # now that we have our rectangle of points, let's compute the width of our new image
        (tl, tr, br, bl) = rect
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
        playing_card = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

        if playing_card.shape[1] > playing_card.shape[0]:
            playing_card = imutils.rotate_bound(playing_card, 90)


        cv2.imshow("transformed card", playing_card)

        key = cv2.waitKey(delay=0)

        if key == ord('n') or key == ord('N'):
            card_count += 1
            # cv2.destroyAllWindows()
        elif key == ord('q') or key == ord('Q'):
            # cv2.destroyAllWindows()
            break

    return playing_card
