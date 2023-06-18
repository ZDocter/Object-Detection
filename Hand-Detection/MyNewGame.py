import cv2
import mediapipe as mp
import time  # 用于检查帧速率
from HandTrackingModule import *

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)  # 调用网络摄像头
detector = HandDetector()
while True:
    success, img = cap.read()
    img = detector.findHnads(img)
    lmlist = detector.findPosition(img)
    if len(lmlist) != 0:
        print('拇指指尖位置：', lmlist[4])

    cTime = time.time()  # 显示当前时间
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Hands detection", img)  # 展示图片
    cv2.waitKey(1)  # 延时
