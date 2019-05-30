from __future__ import print_function
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


def order_points_old(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--new", type=int, default=-1)
args = vars(ap.parse_args())

# loading our input image, converting it to grayscale, and bluring it
image = cv2.imread("example.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# performing edge detection
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# finding contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sorting the contours
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

# loop over the contours individually
for (i, c) in enumerate(cnts):
    if cv2.contourArea(c) < 100:
        continue

    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

    print("Object #{}:".format(i + 1))
    print(box)

    rect = order_points_old(box)

    if args["new"] > 0:
        rect = perspective.order_points(box)

    print(rect.astype("int"))
    print("")

    for ((x, y), color) in zip(rect, colors):
        cv2.circle(image, (int(x), int(y)), 5, color, -1)

    cv2.putText(image, "Object #{}".format(i + 1), (int(rect[0][0] - 15), int(rect[0][1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
