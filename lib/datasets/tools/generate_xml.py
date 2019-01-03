# -*- coding=utf-8 -*-
#------------------------------------
# generate xml from txt label files
# ICDAR format to VOC format
#------------------------------------

import os,sys
import glob
import json
import numpy as np
import xml.etree.cElementTree as ET
import cv2
import math
import io
from check_point import check_point

reload(sys)
sys.setdefaultencoding("utf8")

def labelfromtxt(label_path, clear):
    rboxes = []

    with io.open(label_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        line_num = len(lines)
        for line in lines:
            data = line.split(',')[0:9]
            points_data = map(int,data[0:8])
            #points_data = map(int, data[0:8])
            str_data = str(data[8])
            points= [[points_data[0],points_data[1]],\
            [points_data[2],points_data[3]],[points_data[4],points_data[5]],[points_data[6],points_data[7]]]
            if clear and (str_data[0]=='#' or len(str_data)==2):
                continue
            rboxes.append(points)
    if f:
        f.close()  # 确保文件被关闭
    return rboxes

def generate_imageset(root_dir, filename, setname='trainval.txt'):
    setfilePath = os.path.join(root_dir, setname)
    with open(setfilePath, 'a') as f:
        f.writelines(filename.split('.')[0]+'\n')
    f.close()
    return 0


if __name__ == '__main__':

    # label_dir = "/Users/lairf/Documents/RefineData/OcrTestLabelElse/"
    # image_dir = "/Users/lairf/Documents/RefineData/OcrTestDataElse/"
    # output_dir = "/Users/lairf/Documents/RefineData/xml/"
    images_dir = "./data/ICDAR2015/JPEGImages/"
    labels_dir = "./data/ICDAR2015/Annotationstxt/"
    output_dir = "./data/ICDAR2015/Annotations/"
    imageset_dir = "./data/ICDAR2015/ImageSets/Main"
    global enumerate
    num=0
    in_path = labels_dir +  "*.txt"
    for txt_file in glob.glob(in_path):
        #print txt_file
        rboxes = labelfromtxt(txt_file, clear=1)
        img_path = txt_file.replace(labels_dir,images_dir)
        img_path = img_path.replace('gt_','')
        img_path = img_path.replace('.txt','.jpg')
        print(img_path)
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        imageFileName = img_path.split('/')[-1]
        imagePath = img_path

        generate_imageset(imageset_dir, imageFileName)
        num+=1
        if num%100==0:
            print(num)

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "ICDAR2015"
        ET.SubElement(root, "filename").text = imageFileName

        source = ET.SubElement(root, "source")
        ET.SubElement(source, "dataset").text = "ICDAR2015"
        ET.SubElement(source, "id").text = '0'

        owner = ET.SubElement(root, "owner")
        ET.SubElement(owner, "name").text = "Linkface"

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(channels)

        ET.SubElement(root, "segmented").text = '0'

        #f0 = Res['objects']['ocr'][0]['polygonList']
        for rbox in rboxes:
            # correct the data ensure the range of coordination is [0,width/height)
            x0 = max(0, rbox[0][0])
            y0 = max(0, rbox[0][1])
            x1 = max(0, rbox[1][0])
            y1 = max(0, rbox[1][1])
            x2 = max(0, rbox[2][0])
            y2 = max(0, rbox[2][1])
            x3 = max(0, rbox[3][0])
            y3 = max(0, rbox[3][1])
            
            x0 = min(width-1, x0)
            x1 = min(width-1, x1)
            x2 = min(width-1, x2)
            x3 = min(width-1, x3)
            
            y0 = min(height-1, y0)
            y1 = min(height-1, y1)
            y2 = min(height-1, y2)
            y3 = min(height-1, y3)

            pts = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], np.int32)
            xmin = min(pts[:, 0])
            xmax = max(pts[:, 0])
            ymin = min(pts[:, 1])
            ymax = max(pts[:, 1])

            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = 'text'
            ET.SubElement(obj, "pose").text = 'Left'
            ET.SubElement(obj, "truncated").text = '1'
            ET.SubElement(obj, "difficult").text = '0'
            # ET.SubElement(obj, "content").text = line['attributes']['content']['value']
            
            box = ET.SubElement(obj, "bndbox")
            
            if xmin<0 or ymin<0 or xmax>=width or ymax>=height:
                print('data error1')

            ET.SubElement(box, "xmin").text = str(xmin)
            ET.SubElement(box, "ymin").text = str(ymin)
            ET.SubElement(box, "xmax").text = str(xmax)
            ET.SubElement(box, "ymax").text = str(ymax)

            xmlrbox = ET.SubElement(obj, "rbox")
            rect = cv2.minAreaRect(np.array(rbox))  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            points = cv2.boxPoints(rect)       # points 和 rbox方向不一定一致！！！！！
            img_info = [width, height]
            points = check_point(points, img_info)

            x0 = int(points[0][0])
            y0 = int(points[0][1])
            x1 = int(points[1][0])
            y1 = int(points[1][1])
            x2 = int(points[2][0])
            y2 = int(points[2][1])
            
            if x2<0 or y2<0 or x2>=width or y2>=height:
                print('data error2') 

            ET.SubElement(xmlrbox, "x0").text = str(x0)
            ET.SubElement(xmlrbox, "y0").text = str(y0)
            ET.SubElement(xmlrbox, "x1").text = str(x1)
            ET.SubElement(xmlrbox, "y1").text = str(y1)
            rboxh = math.sqrt((x2-x1)**2+(y2-y1)**2)
            ET.SubElement(xmlrbox, "h").text = str(int(rboxh))

        xmlFileName = imageFileName[:-3] + "xml"
        xmlPath = output_dir + xmlFileName
        xml = ET.ElementTree(root)
        xml.write(xmlPath)
