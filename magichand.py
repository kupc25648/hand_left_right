'''
**Credit
Neural Network from Openpose teams : https://github.com/CMU-Perceptual-Computing-Lab/openpose

Program to press left and right button using hands detection nn from openpose
'''

import cv2
import numpy as np
import pyautogui
import time
pyautogui.FAILSAFE = True
time.sleep(5)


# Import weigths
protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


# Test window
frameWidth = 640
frameHeight = 380
cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,150)

move_thred = 30

# Function to Detect Hands
def findhand(img):
    inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (frameWidth, frameHeight),
                          (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    # fingers point 4,8,12,16,20
    # finger 1
    finger1 = output[0, 4, :, :]
    finger1 = cv2.resize(finger1, (frameWidth, frameHeight))
    minVal1, prob1, minLoc1, point1 = cv2.minMaxLoc(finger1)
    # finger 2
    finger2 = output[0, 8, :, :]
    finger2 = cv2.resize(finger2, (frameWidth, frameHeight))
    minVal2, prob2, minLoc2, point2 = cv2.minMaxLoc(finger2)
    # finger 3
    finger3 = output[0, 12, :, :]
    finger3 = cv2.resize(finger3, (frameWidth, frameHeight))
    minVal3, prob3, minLoc3, point3 = cv2.minMaxLoc(finger3)
    # finger 4
    finger4 = output[0, 16, :, :]
    finger4 = cv2.resize(finger4, (frameWidth, frameHeight))
    minVal4, prob4, minLoc4, point4 = cv2.minMaxLoc(finger4)
    # finger 5
    finger5 = output[0, 20, :, :]
    finger5 = cv2.resize(finger5, (frameWidth, frameHeight))
    minVal5, prob5, minLoc5, point5 = cv2.minMaxLoc(finger5)

    fingers = [prob1,prob2,prob3,prob4,prob5]
    postions = [point1,point2,point3,point4,point5]
    print(fingers)
    print(postions)
    return fingers,postions

# Detect funtion
def Detect(fingers,postions,threshold=0.02):
    finger_name = ['fing1','fing2','fing3','fing4','fing5']
    detect = []
    for i in range(len(fingers)):
        if fingers[i] >= threshold:
            detect.append([finger_name[i],postions[i][0],postions[i][1]])
    return detect

while True:
    success, img = cap.read()
    imgResult = img.copy()
    # detect 1
    fingers, positons = findhand(img)
    detect1 = Detect(fingers,positons)
    cv2.waitKey(50)
    success, img = cap.read()
    imgResult = img.copy()
    # detect 2
    fingers, positons = findhand(img)
    detect2 = Detect(fingers,positons)
    # Direction
    Direction = 0 #(+right,-left)
    count = 0
    for i in range(len(detect2)):
        for j in range(len(detect1)):
            if detect2[i][0] == detect1[j][0]:
                Direction+=(detect2[i][1]-detect1[j][1])
                count += 1
    if count != 0:
        Direction/=count
        if Direction>move_thred:
            print('move right')
            pyautogui.press('right')
        elif Direction<-move_thred:
            print('move left')
            pyautogui.press('left')
    else:
        print('not found')

    print('----------------')
    print(Direction)
    Direction=0
    count = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



