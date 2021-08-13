import cv2
import numpy as np


def perspective_transformation(img, perspectiveImg):
        img_ori = img
        # Read the image, do grayscale, Gaussian blur, dilation, Canny edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        # edged = cv2.Canny(dilate, 75, 200)
        edged = cv2.Canny(dilate, 30, 120, 3)
        cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if imutils.is_cv2() else cnts[1] # determine whether it is OpenCV2 or OpenCV3
        docCnt = None
        # Make sure to find at least one outline
        if len(cnts) > 0:
            # Sort by descending outline size
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            draw_img = cv2.drawContours(img.copy(), cnts, -1, (0, 0, 255), 3)

            for c in cnts:
                # Approximate contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.1 * peri, True)

                if len(approx) == 4:
                    docCnt = approx
                    '''cv2.circle(img, tuple(docCnt[0][0]), 2, (255, 0, 255), 5)
                    cv2.circle(img, tuple(docCnt[1][0]), 2, (255, 0, 255), 5)
                    cv2.circle(img, tuple(docCnt[2][0]), 2, (255, 0, 255), 5)
                    cv2.circle(img, tuple(docCnt[3][0]), 2, (255, 0, 255), 5)'''
                    cv2.line(img_ori, tuple(docCnt[0][0]), tuple(docCnt[1][0]), (255, 0, 255), 5)
                    cv2.line(img_ori, tuple(docCnt[1][0]), tuple(docCnt[2][0]), (255, 0, 255), 5)
                    cv2.line(img_ori, tuple(docCnt[2][0]), tuple(docCnt[3][0]), (255, 0, 255), 5)
                    cv2.line(img_ori, tuple(docCnt[3][0]), tuple(docCnt[0][0]), (255, 0, 255), 5)

                    #cv2.imshow("img_ori", img_ori)
                    break

        # Apply a four-point perspective transform to the original image to obtain a top view of the paper
        if docCnt is not None:
            # srcPoints = np.vstack((docCnt[0][0], docCnt[3][0], docCnt[1][0], docCnt[2][0]))
            srcPoints = np.vstack((docCnt[3][0], docCnt[2][0], docCnt[0][0], docCnt[1][0]))
            srcPoints = np.float32(srcPoints)
            # 目标的像素值大小
            h, w = img.shape[:2]
            # 设置目标画布的大小
            canvasPoints = np.array([[0, 0], [int(w), 0], [0, int(h)], [int(w), int(h)]])
            canvasPoints = np.float32(canvasPoints)

            # 计算转换矩阵
            perspectiveMatrix = cv2.getPerspectiveTransform(srcPoints, canvasPoints)
            perspectiveImg = cv2.warpPerspective(img, perspectiveMatrix, (w, h))
            #cv2.imshow("perspectiveImg/梯形校正文档", perspectiveImg)

        return perspectiveImg

def mouse_select_transformation(img, perspectiveImg):
    print('hello')
