import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import os
import cv2
import math

root_path = './data/ICDAR2015/'
index = 'img_34'
classes = ('__background__',  'text')   
num_classes = 2

def show(im):
    msg = 'press any key to continue'
    cv2.namedWindow(msg, cv2.WINDOW_NORMAL)
    cv2.imshow(msg, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_pascal_annotation(index):
  """
  Load image and bounding boxes info from XML file in the PASCAL VOC
  format.
  """
  filename = os.path.join(root_path, 'Annotations', index + '.xml')
  # print(filename)
  tree = ET.parse(filename)
  objs = tree.findall('object')
  num_objs = len(objs)

  boxes = np.zeros((num_objs, 4), dtype=np.uint16)
  rboxes = np.zeros((num_objs, 5), dtype=np.uint16)
  quadboxes = np.zeros((num_objs, 8), dtype=np.uint16)
  gt_classes = np.zeros((num_objs), dtype=np.int32)
  # regression for rotate box (x0,y0,x1,y1,h)
  gt_rbox = np.zeros((num_objs, 5), dtype=np.int32)
  overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)
  # "Seg" area for pascal is just the box area
  seg_areas = np.zeros((num_objs), dtype=np.float32)

  # Load object bounding boxes into a data frame.
  for ix, obj in enumerate(objs):
    bbox = obj.find('bndbox')
    # Make pixel indexes 0-based
    x1 = float(bbox.find('xmin').text)
    y1 = float(bbox.find('ymin').text)
    x2 = float(bbox.find('xmax').text)
    y2 = float(bbox.find('ymax').text)  
    
    class_to_ind = dict(list(zip(classes, list(range(num_classes)))))
    cls = class_to_ind[obj.find('name').text.lower().strip()]
    boxes[ix, :] = [x1, y1, x2, y2]
    
    # add quadbox object information
    quadbox = obj.find('quadbox')
    quadx0 = float(quadbox.find('quadx0').text)
    quady0 = float(quadbox.find('quady0').text)
    quadx1 = float(quadbox.find('quadx1').text)
    quady1 = float(quadbox.find('quady1').text)
    quadx2 = float(quadbox.find('quadx2').text)
    quady2 = float(quadbox.find('quady2').text)
    quadx3 = float(quadbox.find('quadx3').text)
    quady3 = float(quadbox.find('quady3').text)
    quadboxes[ix, :] = [quadx0,quady0,quadx1,quady1,quadx2,quady2,quadx3,quady3]

    # add rbox object information
    rbox = obj.find('rbox')
    rx0 = float(rbox.find('x0').text)
    ry0 = float(rbox.find('y0').text)
    rx1 = float(rbox.find('x1').text)
    ry1 = float(rbox.find('y1').text)
    rh = float(rbox.find('h').text)
    rboxes[ix, :] = [rx0,ry0,rx1,ry1,rh]
    
    gt_classes[ix] = cls
    overlaps[ix, cls] = 1.0
    seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
  overlaps = scipy.sparse.csr_matrix(overlaps)
  # return gt_roidb
  return {'boxes': boxes,
          'rboxes': rboxes,
          'quadboxes': quadboxes,
          'gt_classes': gt_classes,
          'gt_overlaps': overlaps,
          'flipped': False,
          'seg_areas': seg_areas}


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
    return rect

def showrbox(img, rect, width = 5):
    cv2.line(img, tuple(rect[0].astype(np.int32)),tuple(rect[1].astype(np.int32)), color=(255,0,0), thickness=width)
    cv2.line(img, tuple(rect[1].astype(np.int32)),tuple(rect[2].astype(np.int32)), color=(255,0,0), thickness=width)
    cv2.line(img, tuple(rect[2].astype(np.int32)),tuple(rect[3].astype(np.int32)), color=(255,0,0), thickness=width)
    cv2.line(img, tuple(rect[3].astype(np.int32)),tuple(rect[0].astype(np.int32)), color=(255,0,0), thickness=width)
    return 0

def getplateimage(img, boxes):
    dstshape = np.float32([[0,0],[440,0],[440,140],[0,140]])
    boxes = boxes.astype(np.float32)
    M = cv2.getPerspectiveTransform(boxes, dstshape)
    plate_img = cv2.warpPerspective(img, M, (440,140))
    return plate_img

def img_wrapper(img, plate_img, i=0):
    #warpper_shape = (img.shape[0],img.shape[1],img.shape[2])
    #warpper_img = np.zeros(warpper_shape, np.uint8)
    warpper_img = img
    #warpper_img[0:img.shape[0],0:img.shape[1],:] = img
    warpper_img[plate_img.shape[0]*i:plate_img.shape[0]*(i+1),\
                0:plate_img.shape[1],\
                :] = plate_img
    return warpper_img

if __name__ == '__main__':
  root_path = '/Users/yangyd/linkface/week2/quadplate'
  save_path = '/Users/yangyd/linkface/week2/quadplatevis'
  img_list = os.walk(os.path.join(root_path, 'JPEGImages'))
  for _,_,img_names in img_list:
    for img_name in img_names:
      img_name = str(img_name)
      print img_name
      index = img_name.split('.jpg')[0]
      if index is '':
        continue
      roidb = load_pascal_annotation(index)
      imgname = os.path.join(root_path, 'JPEGImages', index + '.jpg')

      img = cv2.imread(imgname)
      org_img = img.copy()
      # detect text regions
      regions = np.hstack((roidb['boxes'],roidb['rboxes'],roidb['quadboxes']))
      for i in range(len(regions)):
          boxes = regions[i]
          print("Boxes {}".format('text'))
          # show bndbox
          cv2.rectangle(img, (boxes[0].astype(np.int32), boxes[1].astype(np.int32)), \
              (boxes[2].astype(np.int32), boxes[3].astype(np.int32)), color=(255, 255, 0), thickness=5)
          #cv2.circle(img, tuple([boxes[0],boxes[1]]), 10 ,color=(255,255,0),thickness=5)
          # show rbox
          #rect = rbox2rect(boxes[4:9])
          #showrbox(img, rect)
          #cv2.circle(img, tuple([boxes[4],boxes[5]]), 10 ,color=(255,0,0),thickness=5)
          # show quadbox
          showrbox(img, boxes[9:17].reshape([-1,2]))
          #cv2.circle(img, tuple([boxes[9],boxes[10]]), 10 ,color=(255,0,0),thickness=5)
          plate_img = getplateimage(org_img, boxes[9:17].reshape([-1,2]))
          all_img = img_wrapper(img, plate_img, i)
                
          #cv2.line(img, tuple([boxes[4],boxes[5]]),tuple([boxes[6],boxes[7]]), color=(255,0,0))
          #cv2.circle(img, tuple([boxes[6],boxes[7]]), 6 ,color=(255,0,0))
      if not os.path.isdir(save_path):
        os.mkdir(save_path)
      cv2.imwrite(os.path.join(save_path, index + '.jpg' ), all_img)
