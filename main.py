#hand tracking
# main idea
#1. Find hand Landmarks
#2. Get the tip of the index and middle fingers
#3. Check which fingers are up
#4. Only Index Finger : Moving Mode
#5. Convert Coordinates
#6. Smoothen Values
#7. Move Mouse
#8. Both Index and middle fingers are up : Clicking Mode
#9. Find distance between fingers
#10. Click mouse if distance short
#11. Frame Rate
#12. Display

import Handmouse
import os
import sys
sys.path.append("./YOLO_OpenVINO/")

def demo_logo():
    print("\n/*********************************/")
    print("/---------------------------------/\n")
    print("               WELCOME               ")
    print("                Demo                 ")
    print("              Zili Liao              ")
    print("                 ^_^                 ")
    print("\n/---------------------------------/")
    print("/*********************************/\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    demo_logo()
    Handmouse.main_handmouse_process()
    

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
