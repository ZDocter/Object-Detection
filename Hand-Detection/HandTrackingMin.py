import cv2
import mediapipe as mp
import time  # 用于检查帧速率

cap = cv2.VideoCapture(0)  # 调用网络摄像头

mpHamds = mp.solutions.hands
# 静态检测设为False，避免实时检测导致卡顿
hands = mpHamds.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 获取图像转换为RGB格式，因为hands类只接受RGB图像
    results = hands.process(imgRGB)  # 输出结果
    # print(results.multi_hand_landmarks)

    # 检测是否存在目标
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)  # 中心位置
                # if id == 0:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHamds.HAND_CONNECTIONS)  # 绘制单手连线

    cTime = time.time()  # 显示当前时间
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS{int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


    cv2.imshow("Hands detection", img)  # 展示图片
    cv2.waitKey(1)  # 延时
