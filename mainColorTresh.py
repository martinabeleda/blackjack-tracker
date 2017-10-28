def findGesture(frame):
    import cv2
    from color_find_hand import color_find_hand

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


