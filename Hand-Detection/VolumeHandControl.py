import cv2
import time
import mediapipe
import numpy as np
from HandTrackingModule import *
import math


#########################################
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()  # 获取音量范围
minvol, maxvol = volRange[0], volRange[1]
##########################################

########################
wCam, hCam = 640, 480
cTime = 0
pTime = 0
########################

cmp = cv2.VideoCapture(0)  # 检查摄像头
cmp.set(3, wCam)
cmp.set(4, hCam)
volBar = 400
volPer = 0

detector = HandDetector(maxHands=1, detectionCom=0.5)

while True:
    success, img = cmp.read()
    img = detector.findHnads(img)  # 绘制手的轮廓
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # 绘制需要的两个点位
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

        # 绘制两点连线
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)  # 绘制连线中心

        # 计算连线长度
        length = math.hypot(x2 - x1, y2 - y1)
        print(length)

        # 将声音与距离转化为对应量级
        vol = np.interp(length, [50, 300], [minvol, maxvol])
        volBar = np.interp(length, [50, 300], [400, 120])  # 音量条
        volPer = np.interp(length, [50, 300], [0, 100])  # 百分比
        volume.SetMasterVolumeLevel(vol, None)

        if length <= 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)  # 绘制连线中心

    cv2.rectangle(img, (50, 120), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 255, 0), cv2.FILLED)
    cv2.putText(img, f'Voice:{int(volPer)}%', (10, 450), cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 255, 0), 2)
    # 显示FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1,
                (0, 0, 0), 2)
    cv2.imshow('Image', img)  # 显示
    cv2.waitKey(1)
