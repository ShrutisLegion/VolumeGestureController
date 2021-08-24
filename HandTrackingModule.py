import cv2 as cv
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 10, (255, 0, 255), cv.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3,
                   (255, 0, 255), 3)
        cv.imshow("Image", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

# import cv2 as cv
# import mediapipe as mp
# import time

# class handDetector():
#     def __int__(self, mode=False, maxHands=2, detectionCon=0.5, trackingCon=0.5):
#         self.mode = mode
#         self.maxHands = maxHands
#         self.detectionCon = detectionCon
#         self.trackingCon = trackingCon
#         self.mpHands = mp.solutions.hands
#         self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackingCon)
#         self.mpDraw = mp.solutions.drawing_utils
#
#     def findHands(self, frame, draw=True):
#         frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#         self.results = self.hands.process(frameRGB)
#
#         # print(results.multi_hand_landmarks)
#         if self.results.multi_hand_landmarks:
#             for handLms in self.results.multi_hand_landmarks:
#                 if draw:
#                     self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
#
#         return frame

# for id, lm in enumerate(handLms.landmark):
#     # print(id, lms)
#     h, w, ch = frame.shape
#     cx, cy = int(lm.x * w), int(lm.y * h)
#
#     if id == 4:
#         cv.circle(frame, (cx, cy), 20, (123, 123, 0), -1)


# def imp():
#     pTime = 0
#     cTime = 0
#
#     cap = cv.VideoCapture(0)
#     detector = handDetector()
#
#     while cap.isOpened():
#         res, frame = cap.read()
#         frame = detector.findHands(frame)
#
#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime
#
#         cv.putText(frame, str(int(fps)), (20, 70), cv.FONT_ITALIC, 2, (255, 0, 0), 2)
#
#         cv.imshow("Frame", frame)
#
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     imp()
