import cv2
import mediapipe as mp
import time  # 用于检查帧速率


class HandDetector():
    def __init__(self, mode=True, maxHands=2, complexity=1, detectionCom=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCom = detectionCom
        self.trackCon = trackCon

        self.mpHamds = mp.solutions.hands
        # 静态检测设为False，避免实时检测导致卡顿
        self.hands = self.mpHamds.Hands(self.mode, self.maxHands, self.complexity, self.detectionCom, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHnads(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 获取图像转换为RGB格式，因为hands类只接受RGB图像
        self.results = self.hands.process(imgRGB)  # 输出结果
        # print(results.multi_hand_landmarks)

        # 检测是否存在目标
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHamds.HAND_CONNECTIONS)  # 绘制单手连线
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmlist = []

        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # 中心位置
                # print(id, cx, cy)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmlist


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # 调用网络摄像头
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHnads(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()  # 显示当前时间
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS{int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Hands detection", img)  # 展示图片
        cv2.waitKey(1)  # 延时


if __name__ == '__main__':
    main()
