import cv2
from copy import copy
import numpy as np
import os
import json

dict = {}
filesneeded = os.listdir("/home/duanchengqi20/Patch/image/rec2")
files = os.listdir("/home/duanchengqi20/Patch/diff")

def ShapeDetection(img):
    a = -1
    maxx = 0
    maxy = 0
    maxw = 0
    maxh = 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #寻找轮廓点
    for i,obj in enumerate(contours):
        area = cv2.contourArea(obj)  #计算轮廓内区域的面积
        cv2.drawContours(imgContour, obj, -1, (255, 0, 0), 4)  #绘制轮廓线
        perimeter = cv2.arcLength(obj,True)  #计算轮廓周长
        approx = cv2.approxPolyDP(obj,0.02*perimeter,True)  #获取轮廓角点坐标
        CornerNum = len(approx)   #轮廓角点的数量
        x, y, w, h = cv2.boundingRect(approx)  #获取坐标值和宽度、高度

        #轮廓对象分类
        if CornerNum ==3: objType ="triangle"
        elif CornerNum == 4:
            if w==h: objType= "Square"
            else:objType="Rectangle"
        elif CornerNum>4: objType= "Circle"
        else:objType="N"
        if y > 600 or y < 400 or x < 800 or x > 1100:
            continue
        if w*h > a:
            a = copy(w*h)
            maxx = copy(x)
            maxy = copy(y)
            maxw = copy(w)
            maxh = copy(h)
    dict["{}".format(file)] = (maxx, maxy, maxw, maxh)
    cv2.rectangle(imgContour,(maxx,maxy),(maxx+maxw,maxy+maxh),(0,0,255),2)  #绘制边界框
    cv2.imwrite("/home/duanchengqi20/Patch/image3/{}".format(file),imgContour)

for i, file in enumerate(files):
    print(i)
    
    if ".jpg" not in file:
        continue

    path = os.path.join("/home/duanchengqi20/Patch/diff", file)
    img = cv2.imread(path)
    imgContour = img.copy()

    imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  
    imgCanny = cv2.Canny(imgGray,60,60)  
    ShapeDetection(imgCanny)  

with open('/home/duanchengqi20/Patch/loc_person.json', 'w') as f:
        json.dump(dict, f)

