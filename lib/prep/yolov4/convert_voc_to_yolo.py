##
# Python Code is based on this gist, but has been modified for our PolypsSet folder structure:
# Also incorporated ConvertPascalVocToYolo class, so its easier to use in Jupyter
# https://gist.github.com/Amir22010/a99f18ca19112bc7db0872a36a03a1ec
# , which is originally based on Joseph Redmon's Darknet: voc_label.py
# https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
##

import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
from tqdm import tqdm

# train folder will contain images and xml voc annotations
dirs = ['data/train2019']
classes = ['adenomatous', 'hyperplastic']

class ConvertPascalVocToYolo:
    def __init__(self, user_dirs = dirs, user_classes = classes):
        self.m_dirs = user_dirs
        self.m_classes = user_classes
        self.run()

    # TODO: Modify getImagesInDir for valid and test
    # Already have the functions for it

    # Currently getImagesDir works for train2019 dir
    def getImagesInDir(self, dir_path):
        image_list = []
        # print("image dir_path =", dir_path)
        for filename in glob.glob(dir_path + '/*.jpg'):
            # print("filename =", filename)
            image_list.append(filename)

        return image_list

    def convert(self, size, box):
        dw = 1./(size[0])
        dh = 1./(size[1])
        x = (box[0] + box[1])/2.0 - 1
        y = (box[2] + box[3])/2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)

    def convert_annotation(self, dir_path, output_path, image_path):
        basename = os.path.basename(image_path)
        # print("basename =", basename)
        basename_no_ext = os.path.splitext(basename)[0]
        # print("basename_no_ext =", basename_no_ext)

        # print("in dir_path =", dir_path)
        in_file = open(dir_path + '/' + basename_no_ext + '.xml')
        # print("out output_path =", output_path)
        out_file = open(output_path + basename_no_ext + '.txt', 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        # print("w = ", w, "; h =", h)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            # print("cls =", cls)
            if cls not in self.m_classes or int(difficult)==1:
                continue
            cls_id = self.m_classes.index(cls)
            # print("cls_id =", cls_id)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = self.convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    def run(self):
        cwd = getcwd()
        # print("CWD = ", cwd)

        for dir_path in self.m_dirs:
            full_dir_path = cwd + '/' + dir_path
            # print("full_dir_path =", full_dir_path)
            image_base_dir = full_dir_path + '/Image/'
            # print("image_base_dir =", image_base_dir)
            # Annotation/ is PascalVOC
            pascal_voc_path = full_dir_path + '/Annotation/'
            # print("pascal_voc_path =", pascal_voc_path)
            output_path = full_dir_path +'/yolo/'

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            image_paths = self.getImagesInDir(image_base_dir)
            list_file = open(full_dir_path + '.txt', 'w')

            for image_path in tqdm(image_paths):
                list_file.write(image_path + '\n')
                self.convert_annotation(pascal_voc_path, output_path, image_path)
            list_file.close()

            print("Finished processing: " + dir_path)