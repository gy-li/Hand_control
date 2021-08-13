#基础模块
import cv2
import numpy as np
import time
import easyocr

from multiprocessing import Process
from multiprocessing import Manager

#自设模块
import component.HandtrackingModule as htm
import component.perspective as per
import component.Chinese_text as cht
import component.additionModule as adm
from YOLO_OpenVINO.detect import detect

#import test_openvino as text_recognition

# easyOCR模型读取
reader = easyocr.Reader(['ch_sim', 'en'])
coor = np.array([[1,1]])
ft = cht.put_chinese_text('./component/testChinese.TTF')

def hand_mouse(info_dict):

    ##########################
    #wCam, hCam = 640, 480
    #frameR = 100  # Frame Reduction
    #smoothening = 7
    #########################

    #帧数计算，pTime为上一帧时间
    pTime = 0
    frame = 0

    #平滑鼠标移动
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    # 视频抓取
    cap = cv2.VideoCapture(0)
    # 获取视频尺寸
    size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    minx, maxx, miny, maxy = size[0], 0, size[1], 0
    minx_1, maxx_1, miny_1, maxy_1 = size[0], 0, size[1], 0
    #cap.set(3, wCam)
    #cap.set(4, hCam)

    # 检测手部, 默认属性:maxHands = 2, detectionCon = 0.8, trackCon = 0.5
    detector = htm.handDetector(maxHands=2)
    #tr = text_recognition.TextRecognizer()

    # 保存鼠标点击位置 (为鼠标选择梯形校正区域)
    Click = []
    # 保存梯形校正图片
    perspectiveImg = None
    # 保存手指框选区域图片
    img_reco_crop_cir = None
    img_reco_crop_cir_per = None

    # 监听鼠标事件
    def OnMouseAction(event, x, y, flags, param):
        global coor_x, coor_y, coor
        if event == cv2.EVENT_LBUTTONDOWN:
            print("左键点击")
            print("%s" % x, y)
            coor_x, coor_y = x, y
            # cv2.circle(img, (x, y), 15, (255, 0, 255), cv2.FILLED)
            coor_m = [coor_x, coor_y]
            if len(Click) <4:
                Click.append(coor_m)
            else:
                print("4 point have been chosen")
            coor = np.row_stack((coor, coor_m))
        elif event == cv2.EVENT_LBUTTONUP:
            print("%s" % x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("右键点击")
            Click.pop()
        elif flags == cv2.EVENT_FLAG_LBUTTON:
            print("左鍵拖曳")
        elif event == cv2.EVENT_MBUTTONDOWN:
            print("中键点击")

    info_dict["handpose_procss_ready"] = True
    # 读取当前视频图片并进行操作
    while True:

        # 读取视频图片
        success, img_ori = cap.read()
        if success:
            #图片反转，根据当前场景需要
            img = cv2.flip(img_ori, 0)
            img = cv2.flip(img, 1)
            #img = cv2.flip(img_ori, 1)

        else:
            print('No camera detected')
            break
        #框出手并获取当前手位置及手势
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        #r = cv2.selectROI('input', img, False)


        # 获取中指和食指顶尖位置
        if lmList != []:
            x1, y1 = lmList[0][1]['hand_21_point'][8][1:]
            x2, y2 = lmList[0][1]['hand_21_point'][12][1:]
            # print(x1, y1, x2, y2)

            # 检查哪只手指此时竖起
            # 主操作手
            fingers_main = []
            fingers_second = []
            fingers_main = lmList[0][1]["fingersUp"]
            # 副操作手
            if len(lmList) >= 2:
                fingers_second = lmList[1][1]["fingersUp"]
            #print(fingers_main,fingers_second)
            #cv2.rectangle(img, (frameR, frameR ), (wCam - frameR, hCam - frameR),(255, 0, 255), 2)

            # 单食指竖起，模拟鼠标移动，以主操作手模拟鼠标移动，目前关闭鼠标移动操作，仅显示当前手势处于鼠标移动操作，蓝点为当前鼠标位置
            if fingers_main[1:] == [1,0,0,0]:
                '''# 5. Convert Coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))        # print(fingers)
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                # 6. Smoothen Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                # 7. Move Mouse
                autopy.mouse.move(wScr - clocX, clocY)'''
                cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # 食指和中指同时竖起并双指触碰为选择模式(鼠标点击)，此时可框选所需区域，或双手同时选择区域
            if fingers_main[1:] == [1,1,0,0] and fingers_second[1:] == [1,1,0,0]:
                length_main_index_middle, img, lineInfo_main = detector.findDistance(0, 8, 12, img)
                length_second_index_middle, img, lineInfo_second = detector.findDistance(1, 8, 12, img)
                if length_main_index_middle < 50 and length_second_index_middle < 50:
                    cv2.circle(img, (lineInfo_main[4], lineInfo_main[5]), 15, (0, 255, 0), cv2.FILLED)
                    cv2.circle(img, (lineInfo_second[4], lineInfo_second[5]), 15, (0, 255, 0), cv2.FILLED)
                    minx_1, maxx_1, miny_1, maxy_1 = min(minx_1, lineInfo_main[4], lineInfo_second[4]), \
                                             max(maxx_1, lineInfo_main[4], lineInfo_second[4]), \
                                             min(miny_1, lineInfo_main[5], lineInfo_second[5]), \
                                             max(maxy_1, lineInfo_main[5], lineInfo_second[5])
                    if info_dict["Object_pending_state"] == False:
                        info_dict["Object_pic"] = img[miny_1:maxy_1,minx_1:maxx_1,:]
                        info_dict["Object_detect"] = True

            elif fingers_main[1:] == [1,1,0,0]:
                # 计算食指和中指距离
                length_main_index_middle, img, lineInfo_main = detector.findDistance(0, 8, 12, img)
                # 如果距离较短则此时为点击模式
                if length_main_index_middle < 50:
                    cv2.circle(img, (lineInfo_main[4], lineInfo_main[5]),15, (0, 255, 0), cv2.FILLED)
                    minx,maxx,miny,maxy = min(minx, lineInfo_main[4]), max(maxx, lineInfo_main[4]), min(miny,lineInfo_main[5]), max(maxy,lineInfo_main[5])


            if ((maxx-minx)>0) and ((maxy-miny)>0):
                img_reco_crop_cir = img[miny:maxy,minx:maxx,:]
                if img_reco_crop_cir is not None and info_dict["Ocr_pending_state"] == False:
                    #cv2.imshow("img_reco_crop_cir", img_reco_crop_cir)
                    info_dict["Ocr_pic"] = img_reco_crop_cir
                    info_dict["Ocr_info"] = "img_reco_crop_cir"
                    info_dict["Ocr_detect"] = True

            # 主操作手握拳为取消选择图片
            if fingers_main == [0,0,0,0,0]:
                minx, maxx, miny, maxy = 1920, 0, 1080, 0
                img_reco_crop_cir = None


            if fingers_main[1:] == [1,1,1,0]:
                # perspectiveImg = per.perspective_transformation(img_ori, perspectiveImg)
                if perspectiveImg is not None and info_dict["Ocr_pending_state"] == False:
                    info_dict["Ocr_pic"] = perspectiveImg
                    info_dict["Ocr_info"] = "Ori_pic_Oct"
                    info_dict["Ocr_detect"] = True
                elif info_dict["Ocr_pending_state"] == False:
                    info_dict["Ocr_pic"] = img
                    info_dict["Ocr_info"] = "Ori_pic_Oct"
                    info_dict["Ocr_detect"] = True
    
        # 11. Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        # 12. Display
        cv2.namedWindow('Image')
        # 鼠标选择点击区域
        cv2.setMouseCallback('Image', OnMouseAction)
        if len(Click) == 4:
            box_line = adm.order_points(np.array(Click)).astype(int)
            cv2.line(img, box_line[0], box_line[1], (255, 0, 255), 1)
            cv2.line(img, box_line[1], box_line[2], (255, 0, 255), 1)
            cv2.line(img, box_line[2], box_line[3], (255, 0, 255), 1)
            cv2.line(img, box_line[3], box_line[0], (255, 0, 255), 1)

            # 目标的像素值大小
            h, w = 600,420
            # 设置目标画布的大小
            canvasPoints = np.array([[0, 0], [int(w), 0], [int(w), int(h)], [0, int(h)]])
            canvasPoints = np.float32(canvasPoints)

            # 计算转换矩阵
            perspectiveMatrix = cv2.getPerspectiveTransform(adm.order_points(np.array(Click)), canvasPoints)
            perspectiveImg = cv2.warpPerspective(img, perspectiveMatrix, (w, h))
            cv2.imshow("perspectiveImg/梯形校正文档", perspectiveImg)
        else:
            for i in Click:
                cv2.circle(img, i, 5, (0, 255, 0), cv2.FILLED)

        cv2.imshow("Image", img)

        k = cv2.waitKey(1)
        if k == 27:  # 键盘上Esc键的键值
            info_dict["break"] = True
            cv2.destroyAllWindows()
            break

def Ocr_detect(info_dict):
    while True:
        if info_dict["Ocr_pic"] is not None and info_dict["Ocr_detect"]:
            info_dict["Ocr_pending_state"] = True
            start_time = time.time()
            result = reader.readtext(info_dict["Ocr_pic"], decoder='greedy', batch_size=512)
            cv2.imwrite('test.png', info_dict["Ocr_pic"], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            text_img = info_dict["Ocr_pic"]
            for i in result:
                text_img = ft.draw_text(text_img, (int(i[0][0][0]), int(i[0][0][1])), i[1], 20, (255, 0, 0))

            f = open('text.txt', 'w', encoding='utf-8')
            f.write(str(result))
            f.close()
            info_dict["Ocr_detect"] = False
            cv2.imshow(info_dict["Ocr_info"], text_img)
            print(time.time()-start_time)
            info_dict["Ocr_pending_state"] = False
            k = cv2.waitKey(1)
            if k == 27:  # 键盘上Esc键的键值
                cv2.destroyAllWindows()
                break
        if info_dict["Object_pic"] is not None and info_dict["Object_detect"]:
            info_dict["Object_pending_state"] = True
            detect(info_dict["Object_pic"])
            info_dict["Object_detect"] = False
            info_dict["Object_pending_state"] = False


def main_handmouse_process():

    print("\n/---------------------- main_handmouse_process--- ------------------------/\n")
    print("\n/------------------------------------------------------------------------/\n")
    print(" loading Intel_hand_control demo ...")
    global_info_dict = Manager().dict()# 多进程共享字典初始化：用于多进程间的 key：value 操作
    global_info_dict["handpose_procss_ready"] = False # 进程间的开启同步信号
    global_info_dict["break"] = False # 进程间的退出同步信号

    global_info_dict["Ocr_pic"] = None # 文字识别图片
    global_info_dict["Ocr_detect"] = False
    global_info_dict["Ocr_pending_state"] = False # 程序是否在进行文字识别
    global_info_dict["Ocr_info"] = ""

    global_info_dict["Object_pic"] = False
    global_info_dict["Object_detect"] = False
    global_info_dict["Object_pending_state"] = False

    print(" multiprocessing dict key:\n")
    for key_ in global_info_dict.keys():
        print( " -> ",key_)
    print()

    #-------------------------------------------------- 初始化各进程
    process_list = []
    t = Process(target=hand_mouse,args=(global_info_dict,))
    process_list.append(t)

    t = Process(target=Ocr_detect,args=(global_info_dict,)) # 文字识别
    process_list.append(t)

    #t = Process(target=Object_detect,args=(global_info_dict,))
    #process_list.append(t)

    for i in range(len(process_list)):
        process_list[i].start()

    for i in range(len(process_list)):
        process_list[i].join()# 设置主线程等待子线程结束

    del process_list