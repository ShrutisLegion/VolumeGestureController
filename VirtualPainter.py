import numpy as np
import cv2 as cv
import mediapipe as mp
import HandTrackingModule as htm
import time
import os

##########################
height = 720
width = 1280
reqcolor = (255, 0, 255)
xp, yp = 0, 0
brushsize = 15
erasersize = 50
##########################

cap = cv.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = htm.handDetector(detectionCon=0.7)

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)

overLaylist = []

for imgPath in myList:
    image = cv.imread(f'{folderPath}/{imgPath}')
    overLaylist.append(image)

header = overLaylist[0]

imgcanvas = np.zeros((720, 1280, 3), np.uint8)

while cap.isOpened():
    success, img = cap.read()
    img = cv.flip(img, 1)

    # Find hand landmarks
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        # print(lmlist)

        # tip of index and middle finger
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        # check the fingers which are up
        fingers = detector.fingersUp()
        # print(fingers)

        # two finger up means selection of a brush/eraser
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("selection mode")
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overLaylist[0]
                    reqcolor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overLaylist[1]
                    reqcolor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overLaylist[2]
                    reqcolor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overLaylist[3]
                    reqcolor = (0, 0, 0)

            cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), reqcolor, -1)

        elif fingers[1] and fingers[2] == False:
            print("drawing mode")
            cv.circle(img, (x1, y1), 15, reqcolor, -1)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if reqcolor == (0, 0, 0):
                cv.line(img, (xp, yp), (x1, y1), reqcolor, erasersize)
                cv.line(imgcanvas, (xp, yp), (x1, y1), reqcolor, erasersize)
            else:
                cv.line(img, (xp, yp), (x1, y1), reqcolor, brushsize)
                cv.line(imgcanvas, (xp, yp), (x1, y1), reqcolor, brushsize)
            xp, yp = x1, y1

    imgGray = cv.cvtColor(imgcanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgcanvas)

    img[0:125, 0:1280] = header
    # img = cv.addWeighted(img, 0.5, imgcanvas, 0.5, 0)
    cv.imshow("Image", img)
    # cv.imshow("Screen", imgcanvas)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
