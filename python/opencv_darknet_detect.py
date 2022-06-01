import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from PIL import Image

from absl import app, flags, logging

import random
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder

def get_filepaths(basepath, remove_ext=False):
	files = []
	filenames = []
	for filename in os.listdir(basepath):
		if remove_ext is True:
			file_name, file_ext = filename.split(".")
			filepath = os.path.join(basepath, file_name)
			files.append(filepath)
			filenames.append(file_name)
		else:
			filepath = os.path.join(basepath, filename)
			files.append(filepath)
			filenames.append(filename)
	return files, filenames

def read_img(img_path, flag = cv2.IMREAD_COLOR):
	# (height, width, 3)
	image = cv2.imread(img_path, flag)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image

def get_images(image_group, size, flag = cv2.IMREAD_COLOR):
	images = []
	for image_path in image_group:
		image = read_img(image_path, flag)
		resized_img = cv2.resize(image, (size, size))
		images.append(resized_img)
	return images

def get_xml_label_names(xml_files):
    label_names = []
    for xml_file in xml_files:
        filepath, file_ext = xml_file.split(".")
        if file_ext == "xml":
            train_y_tree = ET.parse(xml_file)
            train_y_root = train_y_tree.getroot()
            if train_y_root.find("object") != None:
                train_y_object = train_y_root.find("object")
                train_y_polyp_name = train_y_object.find("name").text
            else:
                train_y_polyp_name = "Not Specified"
            label_names.append(train_y_polyp_name)
        else:
            print("Ignoring .jpg file")
    return label_names

def cv_yolo_detect(_argv):
    # Hyperplastic Polyp Image 19 from video 3
    ADENOMATOUS = 0
    HYPERPLASTIC = 1
    classes = ["adenomatous", "hyperplastic"]

    # Read Darknet DNN M
    configPath = "cfg/yolov4-custom.cfg"
    weightsPath = "../trained_weights/yolov4-custom_best.weights"
    yolo_net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    layer_names = yolo_net.getLayerNames()
    # print("len(layer_names) =", len(layer_names))
    # print("layer_names =", layer_names)
    # for layer_i in yolo_net.getUnconnectedOutLayers():
    #     print("layer_i-1 =", layer_i-1)
    #     print("layer_names[layer_i-1] =", layer_names[layer_i-1]) # -1 to prevent out of range; 1D since binary clf;dtc
        # print("layer_names[layer_i[0] - 1] =", layer_names[layer_i[0] - 1])
    # updating layer_names to be just unconnected output layers
    layer_names = [layer_names[layer_i-1] for layer_i in yolo_net.getUnconnectedOutLayers()]

    # Load Labels from obj.names
    labelsPath = "data/obj.names"
    POLYP_LABELS = open(labelsPath).read().strip().split("\n")

    # Preprocessing Input Images: get_filepaths picks up both jpg and xml
    patient_id = 5
    base_img_path = "data/PolypsSet/test/image/" + str(patient_id)

    patient_polyp_img_filepaths, patient_polyp_img_filenames = get_filepaths(base_img_path)
    patient_polyp_images = get_images(patient_polyp_img_filepaths, 416)
    # randomly select one polyp frame from patient ID's video set
    p_id_frame_n = random.randint(0, len(patient_polyp_images)-1)
    # p_id_frame_n = 19 # first hyperplastic polyp detection success
    print("p_id_frame_n =", p_id_frame_n)

    base_label_path = "data/PolypsSet/test/Annotation/" + str(patient_id)

    patient_polyp_label_filepaths, patient_polyp_label_filenames = get_filepaths(base_label_path)
    patient_polyp_labels = get_xml_label_names(patient_polyp_label_filepaths)
    label_enc = LabelEncoder()
    patient_polyp_labels = label_enc.fit_transform(patient_polyp_labels)

    (height, width) = patient_polyp_images[p_id_frame_n].shape[:2]
    blob = cv2.dnn.blobFromImage(patient_polyp_images[p_id_frame_n], 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    # after passing in unconnected output layer names, it makes each detection a list
    layerOutputs = yolo_net.forward(layer_names)
    # layerOutputs = yolo_net.forward()
    # print("layerOutputs = ", layerOutputs)

    # Initializing for getting Box Coordinates, Confidences, ClassID 
    boxes = []
    confidences = []
    classIDs = []
    # These two thresholds work well for patient ID 3. No pred for P ID 1, 4.
    conf_threshold = 0.5 # Conf Score threshold
    nms_threshold = 0.4 # IOU threshold

    # Get Confidence:
    # layerOutputs contains a huge 2D array of floats, we need our coordinates
    # to be drawn as bounding boxes, classID, and confidence scores of each detection
    # for each detection from each output layer get the confidence, class id, bounding
    # box params and ignor weak detections (confidence < 0.5)
    for output in layerOutputs:
        # print("len(output) =", len(output))
        for detection in output:
            # print("len(detection) =", len(detection))
            scores = detection[5:] # skip first 5 detections
            classID = np.argmax(scores)
            confidence = scores[classID] # get index of highest score; best prediction

            # compare best prediction against our threshold; if confidence is more, then draw bbox
            if confidence > conf_threshold:
                print("confidence =", confidence)
                print("scores =", scores)
                # get 0 up to but not including 4; 0 to 3 detections for the box
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)


    # Non Maximum Suppression (NMS)
    # NOTE: If done incorrectly object detection will be slow.
    # We need to use our boxes, confidences calculated from previous step
    # Our model returns more than one predictions, more than one boxes are
    # present to a single object. NMS returns the single best bounding box
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw Bounding Boxes; Get our Detector Running Now; Adenomatous Polyp Count; Hyperplastic Polyp Count
    apc = 0
    hpc = 0

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            if POLYP_LABELS[classIDs[i]] == classes[ADENOMATOUS]:
                apc += 1
                color = (255, 100, 0) # RGB will be Green
                cv2.rectangle(patient_polyp_images[p_id_frame_n], (x, y), (x + w, y + h), color, 2)
                text = "{}".format(POLYP_LABELS[classIDs[i]])
                cv2.putText(patient_polyp_images[p_id_frame_n], text, (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            elif POLYP_LABELS[classIDs[i]] == classes[HYPERPLASTIC]:
                hpc += 1
                color = (0, 125, 125) # RGB will be Blue
                cv2.rectangle(patient_polyp_images[p_id_frame_n], (x, y), (x + w, y + h), color, 2)
                text = "{}".format(POLYP_LABELS[classIDs[i]])
                cv2.putText(patient_polyp_images[p_id_frame_n], text, (x + w, y + h),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


    text1 = "No. of Adenomatous Polyps in image: " + str(apc)
    text2 = "No. of Hyperplastic Polyps in image: " + str(hpc)
    color1 = (0, 255, 0)
    color2 = (0, 0, 255)

    # cv2.putText(patient_polyp_images[p_id_frame_n],  text1, (2, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 2)
    # cv2.putText(patient_polyp_images[p_id_frame_n],  text2, (2, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)

    image = Image.fromarray(patient_polyp_images[p_id_frame_n].astype(np.uint8))

    image.show()
    patient_polyp_images[p_id_frame_n] = cv2.cvtColor(patient_polyp_images[p_id_frame_n], cv2.COLOR_BGR2RGB)

    save_dst="cv_yolov4/trained_7000_orig/dtc_images"
    apc_hpc_count = "apc_" + str(apc) + "_hpc_" + str(hpc)

    if not os.path.exists(save_dst):
        os.makedirs(save_dst)

    save_clf_polyp_file = "{}/dtc_polyp_{}_{}.jpg".format(save_dst, apc_hpc_count, p_id_frame_n)
    cv2.imwrite(save_clf_polyp_file, patient_polyp_images[p_id_frame_n])

if __name__ == "__main__":
    try:
        app.run(cv_yolo_detect)
    except SystemExit:
        pass
