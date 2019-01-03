# -*- coding=utf-8 -*-
#------------------------------------
# generate xml from image file lists
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
import shutil
from check_point import check_point

reload(sys)
sys.setdefaultencoding("utf8")

def labelfromtxt(label_path, clear):
    rboxes = []
    keys = []

    with io.open(label_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        line_num = len(lines)
        for line in lines:
            data = line.split(',')[0:9]
            points_data = map(float, data[0:8])
            points_data = map(int, points_data)
            str_data = str(data[8])
            points= [[points_data[0],points_data[1]],\
            [points_data[2],points_data[3]],[points_data[4],points_data[5]],[points_data[6],points_data[7]]]
            if clear and (str_data[0]=='#' or len(str_data)==2):
                continue
            rboxes.append(points)
            keys.append(str_data)
    if f:
        f.close()
    return rboxes, keys

def generate_imageset(root_dir, filename, setname='trainval.txt'):
    setfilePath = os.path.join(root_dir, setname)
    with open(setfilePath, 'a') as f:
        f.writelines(filename.split('.')[0]+'\n')
    f.close()
    return 0


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        data_set = 'VPNewText'
        set_name = 'trainval'
        base_path = '/home/deepfinch/xialy_frcnn/data/VPNewText/'
        input_data_path = '/home/deepfinch/xialy_frcnn/data/VPNewText/train/'
    else:
        data_set = sys.argv[1]  # plate
        set_name = sys.argv[2]  # trainval
        base_path = sys.argv[3] # '/data/plate/faster_rcnn_detect/quadplate/'
        input_data_path = sys.argv[4] # '/data/plate/east_detect/train/image/'

    images_dir = base_path + "JPEGImages/"
    output_dir = base_path + "Annotations/"
    imageset_dir = base_path + "ImageSets/Main"
    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(imageset_dir):
        os.makedirs(imageset_dir)

    global enumerate
    num=0
    in_path = input_data_path +  "*.txt"
    for txt_file in glob.glob(in_path):
        print(txt_file)
        img_path = txt_file[:-4] + ".jpg"
        if not os.path.exists(img_path):
            continue
        print(img_path)
        rboxes, keys = labelfromtxt(txt_file, clear=1)

        img = cv2.imread(img_path)
        height, width, channels = img.shape
        img_info = [width, height]
        image_file_name = img_path.split('/')[-1]
        dst_image_path = images_dir + image_file_name

        num+=1
        if num%100==0:
            print(num)

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = data_set
        ET.SubElement(root, "filename").text = image_file_name

        source = ET.SubElement(root, "source")
        ET.SubElement(source, "dataset").text = data_set
        ET.SubElement(source, "id").text = '0'

        owner = ET.SubElement(root, "owner")
        ET.SubElement(owner, "name").text = "Deepfinch"

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(channels)

        ET.SubElement(root, "segmented").text = '0'

        #f0 = Res['objects']['ocr'][0]['polygonList']
        for rbox, key  in zip(rboxes, keys):
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
            ET.SubElement(obj, "name").text = key #'text'
            ET.SubElement(obj, "pose").text = 'Left'
            ET.SubElement(obj, "truncated").text = '1'
            ET.SubElement(obj, "difficult").text = '0'
            # ET.SubElement(obj, "content").text = line['attributes']['content']['value']

            # add bndbox
            box = ET.SubElement(obj, "bndbox")
            if xmin<0 or ymin<0 or xmax>=width or ymax>=height:
                print('data error1')
            ET.SubElement(box, "xmin").text = str(xmin)
            ET.SubElement(box, "ymin").text = str(ymin)
            ET.SubElement(box, "xmax").text = str(xmax)
            ET.SubElement(box, "ymax").text = str(ymax)

            # add quadbox
            quadbox = ET.SubElement(obj, "quadbox")
            quad_pts = check_point(pts, img_info)
            ET.SubElement(quadbox, "quadx0").text = str(int(quad_pts[0][0]))
            ET.SubElement(quadbox, "quady0").text = str(int(quad_pts[0][1]))
            ET.SubElement(quadbox, "quadx1").text = str(int(quad_pts[1][0]))
            ET.SubElement(quadbox, "quady1").text = str(int(quad_pts[1][1]))
            ET.SubElement(quadbox, "quadx2").text = str(int(quad_pts[2][0]))
            ET.SubElement(quadbox, "quady2").text = str(int(quad_pts[2][1]))
            ET.SubElement(quadbox, "quadx3").text = str(int(quad_pts[3][0]))
            ET.SubElement(quadbox, "quady3").text = str(int(quad_pts[3][1]))

            # add rbox
            xmlrbox = ET.SubElement(obj, "rbox")
            rect = cv2.minAreaRect(np.array(rbox))  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            points = cv2.boxPoints(rect)       # points 和 rbox方向不一定一致！！！！！
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

            if set_name == 'All':
                import random
                if random.randint(0, 9) == 0:
                    generate_imageset(imageset_dir, image_file_name, 'test.txt')
                else:
                    generate_imageset(imageset_dir, image_file_name, 'trainval.txt')
            else:
                generate_imageset(imageset_dir, image_file_name, set_name + '.txt')
            shutil.copyfile(img_path, dst_image_path)

        xmlFileName = image_file_name[:-3] + "xml"
        xmlPath = output_dir + xmlFileName
        xml = ET.ElementTree(root)
        xml.write(xmlPath)
