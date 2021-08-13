import cv2
import mediapipe as mp
import time
import math


def fingersUp(tem_hand_list):
    fingers = [0, 0, 0, 0, 0]
    tipIds = [4, 8, 12, 16, 20]
    # Thumb
    if tem_hand_list == []:
        return [-1, -1, -1, -1, -1]
    if tem_hand_list[4][1] - tem_hand_list[0][1] > 0:
        if tem_hand_list[4][1] > tem_hand_list[4 - 1][1]:
            fingers[0] = 1
        else:
            fingers[0] = 0
    else:
        if tem_hand_list[4][1] < tem_hand_list[4 - 1][1]:
            fingers[0] = 1
        else:
            fingers[0] = 0
    # 4 Fingers
    for id in range(1, 5):
        if tem_hand_list[tipIds[id]][2] < tem_hand_list[tipIds[id] - 1][2] < \
                tem_hand_list[tipIds[id] - 2][2] < tem_hand_list[tipIds[id] - 3][2] or \
                tem_hand_list[tipIds[id]][2] > tem_hand_list[tipIds[id] - 1][2] > \
                tem_hand_list[tipIds[id] - 2][2] > tem_hand_list[tipIds[id] - 3][2]:
            fingers[id] = 1
        else:
            fingers[id] = 0
    return fingers

def judge_click_stabel(img, handpose_list, charge_cycle_step=32):
    flag_click_stable = True
    for i in range(len(handpose_list)):
        _, _, _, dict_ = handpose_list[i]
        id_ = dict_["id"]
        click_cnt_ = dict_["click_cnt"]
        pt_ = dict_["choose_pt"]
        if click_cnt_ > 0:
            # print("double_en_pts --->>> id : {}, click_cnt : <{}> , pt : {}".format(id_,click_cnt_,pt_))
            # 绘制稳定充电环
            # 充电环时间控制
            charge_cycle_step = charge_cycle_step  # 充电步长越大，触发时间越短
            fill_cnt = int(click_cnt_ * charge_cycle_step)
            if fill_cnt < 360:
                cv2.ellipse(img, pt_, (16, 16), 0, 0, fill_cnt, (255, 255, 0), 2)
            else:
                cv2.ellipse(img, pt_, (16, 16), 0, 0, fill_cnt, (0, 150, 255), 4)
            # 充电环未充满，置为 False
            if fill_cnt < 360:
                flag_click_stable = False
        else:
            flag_click_stable = False
    return flag_click_stable

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.6, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20] # thumb,index,middle,ring,pinky

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            if draw:
                for handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img



    def findPosition(self, img, handNo = 0, draw = True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            hand_id = 0
            for handLms in self.results.multi_hand_landmarks:
                #myHand = self.results.multi_hand_landmarks[handNo]
                myHand = handLms
                temlist = []
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    #print(id, cx, cy)

                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                        cv2.putText(img, str(int(id)), (cx + 10, cy + 10), cv2.FONT_HERSHEY_PLAIN,
                                    1, (0, 0, 255), 2)
                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)
                    bbox = xmin, ymin, xmax, ymax

                    temlist.append([id, cx, cy])
                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 255*hand_id), 2)
                    xList = []
                    yList = []
                #判断手指是否抬起
                fingersUp_state = [0,0,0,0,0]
                fingersUp_state = fingersUp(temlist)
                #判断手势是否点击
                click_state = False
                
                self.lmList.append((hand_id,{"fingersUp":fingersUp_state,"hand_21_point":temlist,"click":click_state,"click_cnt":False}))
                hand_id += 1

            #if draw:
                #cv2.putText(img, 'Mouse_controling', (self.lmList[0][1][0][1] + 10, self.lmList[0][1][0][2] + 10), cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 255), 2)

        return self.lmList, bbox

    def findDistance(self, hand_id, p1, p2, img, draw=True, r = 15, t = 3):

        x1, y1 = self.lmList[hand_id][1]["hand_21_point"][p1][1:]
        x2, y2 = self.lmList[hand_id][1]["hand_21_point"][p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

    def draw_click_lines(self, img, img_reco_crop_cir,hand_lines_dict,flag_click_stable,draw = True):
            
        return img_reco_crop_cir

    

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detector = handDetector(maxHands=2)
    while True:
        # 读取视频图片
        success, img_ori = cap.read()
        img = detector.findHands(img_ori)
        lmList, bbox = detector.findPosition(img)
        if lmList !=[]:
            print(lmList[0][1]['hand_21_point'])
        cv2.imshow("Image", img)

        k = cv2.waitKey(1)
        if k == 27:  # 键盘上Esc键的键值
            cv2.destroyAllWindows()
            break