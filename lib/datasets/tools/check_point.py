from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
from operator import itemgetter

def rect2rbox(rect):
    rbox = np.zeros(5)
    x1 = rect[0][0]
    y1 = rect[0][1]
    x2 = rect[1][0]
    y2 = rect[1][1]
    x3 = rect[2][0]
    y3 = rect[2][1]
    h = math.sqrt((y3-y2)**2+(x3-x2)**2)
    rbox = [x1,y1,x2,y2,h]
    return np.array(rbox)

def rbox2rect(rbox):
    rect = np.zeros((4,2))
    x1 = float(rbox[0])
    y1 = float(rbox[1])
    x2 = float(rbox[2])
    y2 = float(rbox[3])
    h = float(rbox[4])
    if abs(x2-x1) <= 1:
        if (y2-y1)<0:
            theta = -math.pi/2
        else:
            theta = math.pi/2
    else:
        theta = math.atan((y2-y1)/(x2-x1))
    dx = h * math.sin(theta)
    dy = h * math.cos(theta)
    
    x3 = x2 - dx
    y3 = y2 + dy
    x4 = x1 - dx
    y4 = y1 + dy

    rect[0] = [x1, y1]
    rect[1] = [x2, y2]
    rect[2] = [x3, y3]
    rect[3] = [x4, y4]
    return np.array(rect)

def check_point(pts,img_info):
    width = img_info[0]
    height = img_info[1]

    #check the boundary
    pts[:,0] = np.minimum(np.maximum(pts[:, 0], 1), width- 1)
    pts[:,1] = np.minimum(np.maximum(pts[:, 1], 1), height-1)

    #check x0
    re = np.where(pts[:,0] == np.min(pts[:,0]))
    if re[0].shape[0] > 1:
        if pts[re[0][0],1] < pts[re[0][1],1]:
            y0 = pts[re[0][0], 1]
            x0 = pts[re[0][0], 0]

        else:
            y0 = pts[re[0][1], 1]
            x0 = pts[re[0][1], 0]
    else:
        x0 = pts[re[0][0],0]
        y0 = pts[re[0][0],1]

    sort_gradient = []
    #get the gredient of x0 with the other points
    for i in range(0, 4):
        if pts[i,0]!= x0 or pts[i,1] != y0:
            sort_gradient.append([get_gradient(pts[i], [x0,y0]), i])
    sort_gradient = sorted(sort_gradient,key=itemgetter(0))

    x2 = pts[sort_gradient[1][1]][0]
    y2 = pts[sort_gradient[1][1]][1]

    b = y2- sort_gradient[1][0]* x2
    #insure x1 and x3
    if pts[sort_gradient[0][1]][0]*sort_gradient[1][0]+ b- pts[sort_gradient[0][1]][1]> 0:
        x1 = pts[sort_gradient[0][1]][0]
        y1 = pts[sort_gradient[0][1]][1]
        x3 = pts[sort_gradient[2][1]][0]
        y3 = pts[sort_gradient[2][1]][1]
    else:
        x3 = pts[sort_gradient[0][1]][0]
        y3 = pts[sort_gradient[0][1]][1]
        x1 = pts[sort_gradient[2][1]][0]
        y1 = pts[sort_gradient[2][1]][1]

    #compute gredient of diagonal
    check_main_diagonal = []
    check_main_diagonal.append([get_gradient([x0, y0], [x2, y2]),0])
    check_main_diagonal.append([get_gradient([x1, y1], [x3, y3]),1])
    check_main_diagonal = sorted(check_main_diagonal,key=itemgetter(0))

    if check_main_diagonal[1][1] == 1:
        main_diagnoal = np.array([[x1, y1], [x3, y3]])
        counter_diagnoal = np.array([[x0, y0], [x2, y2]])
        out_put = sorted(main_diagnoal, key=itemgetter(0))
        x0 = out_put[0][0]
        y0 = out_put[0][1]
        x2 = out_put[1][0]
        y2 = out_put[1][1]
        k = get_gradient([x0,y0],[x2,y2])
        b = y0 - k * x0
        if counter_diagnoal[0][0]* np.float(k) + b - counter_diagnoal[0][1]> 0:
            x3 = counter_diagnoal[1][0]
            y3 = counter_diagnoal[1][1]
            x1 = counter_diagnoal[0][0]
            y1 = counter_diagnoal[0][1]
        else:
            x3 = counter_diagnoal[0][0]
            y3 = counter_diagnoal[0][1]
            x1 = counter_diagnoal[1][0]
            y1 = counter_diagnoal[1][1]
    else:
        main_diagnoal = np.array([[x0, y0], [x2, y2]])
        counter_diagnoal = np.array([[x1, y1], [x3, y3]])
        out_put = sorted(main_diagnoal, key=itemgetter(0))
        x0 = out_put[0][0]
        y0 = out_put[0][1]
        x2 = out_put[1][0]
        y2 = out_put[1][1]
        k = get_gradient([x0, y0], [x2, y2])
        b = y0 - k* x0
        if counter_diagnoal[0][0] * k + b - counter_diagnoal[0][1] > 0:
            x3 = counter_diagnoal[1][0]
            y3 = counter_diagnoal[1][1]
            x1 = counter_diagnoal[0][0]
            y1 = counter_diagnoal[0][1]
        else:
            x3 = counter_diagnoal[0][0]
            y3 = counter_diagnoal[0][1]
            x1 = counter_diagnoal[1][0]
            y1 = counter_diagnoal[1][1]
    #the final output
    pts = np.array([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])
    return pts

def get_gradient(point1,point2):
    if point1[0] == point2[0]:
        if point1[1] > point2[1]:
            return 1e8 #float('-inf')
        else:
            return -1e8
    else:
        return float(point1[1]- point2[1])/(point1[0]- point2[0])

if __name__ =='__main__':
    pts = np.array([[100.,500.],[500.,100.],[900.,500.],[500,900.]])
    print(pts)
    print(check_point(pts,img_info=[1280,1280]))