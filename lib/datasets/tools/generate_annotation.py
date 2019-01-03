# coding=utf-8
import os,sys
import glob
import json
import xml.etree.cElementTree as ET

reload(sys) 
sys.setdefaultencoding("utf8")

# label_dir = "/Users/lairf/Documents/RefineData/OcrTestLabelElse/"
# image_dir = "/Users/lairf/Documents/RefineData/OcrTestDataElse/"
# output_dir = "/Users/lairf/Documents/RefineData/xml/"

label_dir = "/data/linkface/OcrData/VechileLicense2017/label/"
image_dir = "/data/linkface/OcrData/VechileLicense2017/Vechile2017/JPEGImages/"
output_dir = "/data/linkface/OcrData/VechileLicense2017/Vechile2017/Annotations/"

in_path = label_dir +  "*.json"
for json_file in glob.glob(in_path):
    with open(json_file) as file:
        text = file.read().decode("utf-8").strip()
        data = json.loads(text)
        imageFileName = data['image']['rawFilename']
        imagePath = image_dir + imageFileName
        lines = data['objects']['ocr']

        if os.path.exists(imagePath) and len(lines) >= 10:
            print imageFileName

            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = "Vechile2017"
            ET.SubElement(root, "filename").text = imageFileName

            source = ET.SubElement(root, "source")
            ET.SubElement(source, "dataset").text = "Vechile License 2017"
            ET.SubElement(source, "id").text = data['image']['id']
            

            owner = ET.SubElement(root, "owner")
            ET.SubElement(owner, "name").text = "Linkface"

            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(data['image']['width'])
            ET.SubElement(size, "height").text = str(data['image']['height'])
            ET.SubElement(size, "depth").text = '3'

            ET.SubElement(root, "segmented").text = '0'

            for line in lines:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = 'text'
                ET.SubElement(obj, "pose").text = 'Left'
                ET.SubElement(obj, "truncated").text = '1'
                ET.SubElement(obj, "difficult").text = '0'
                # ET.SubElement(obj, "content").text = line['attributes']['content']['value']

                box = ET.SubElement(obj, "bndbox")
                ET.SubElement(box, "xmin").text = str(int(line['position']['left']))
                ET.SubElement(box, "ymin").text = str(int(line['position']['top']))
                ET.SubElement(box, "xmax").text = str(int(line['position']['right']))
                ET.SubElement(box, "ymax").text = str(int(line['position']['bottom']))

            xmlFileName = imageFileName[:-3] + "xml"
            xmlPath = output_dir + xmlFileName
            xml = ET.ElementTree(root)
            xml.write(xmlPath)

