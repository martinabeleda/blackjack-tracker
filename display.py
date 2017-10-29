import cv2

""" This module containts constants and functions for display """

# Drawing
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
CYAN = (255,255,0)
MAGENTA = (255,0,255)


def regions(image, surface_obj):
    x_offset = 10
    y_offset = 60
    shadow_offset = 2
    cv2.line(image, (surface_obj.dealer_region[1] + shadow_offset, 0),
             (surface_obj.dealer_region[1], int(surface_obj.height)), (0, 0, 0), 2)
    cv2.line(image, (surface_obj.dealer_region[1], 0),
             (surface_obj.dealer_region[1], int(surface_obj.height)), (255, 255, 255), 2)
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
