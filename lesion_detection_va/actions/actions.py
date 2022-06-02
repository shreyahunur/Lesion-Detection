# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

# NOTE: cd into actions folder, then run "rasa run actions", so
# it can pick up custom modules in core directory

# This is a simple example for a custom action which utters "Hello World!"
import os
import cv2
import time
import random
import numpy as np
import pandas as pd

# from typing import Any, Text, Dict, List
from rasa_sdk import Action #, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

import matplotlib.pyplot as plt

from PIL import Image

# Keras CNN Polyp Classifer
from tensorflow.keras.models import load_model
import h5py
from tensorflow.keras import __version__ as keras_version

import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder

#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

class ActionRunCNN(Action):
	def name(self):
		return "action_run_cnn"

	def run(self, dispatcher, tracker, domain):
		dispatcher.utter_message(text="Running Polyp Classifier")
		return []
	
class ActionRunImageClassification(Action):
	def name(self):
		return "action_run_image_classification"

	def get_filepaths(self, basepath, remove_ext=False):
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

	def read_img(self, img_path, flag = cv2.IMREAD_COLOR):
		# (height, width, 3)
		image = cv2.imread(img_path, flag)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image

	def get_images(self, image_group, size, flag = cv2.IMREAD_COLOR):
		images = []
		for image_path in image_group:
			image = self.read_img(image_path, flag)
			resized_img = cv2.resize(image, (size, size))
			images.append(resized_img)
		return images

	def get_txt_label_names(self, txt_files):
		label_names = []
		for txt_file in txt_files:
			file = open(txt_file, 'r+')
			polyp_name = file.readline()
			label_names.append(polyp_name)
			file.close()
		return label_names

	def get_xml_label_names(self, xml_files):
		label_names = []
		for xml_file in xml_files:
			train_y_tree = ET.parse(xml_file)
			train_y_root = train_y_tree.getroot()
			if train_y_root.find("object") != None:
				train_y_object = train_y_root.find("object")
				train_y_polyp_name = train_y_object.find("name").text
			else:
				train_y_polyp_name = "Not Specified"
			label_names.append(train_y_polyp_name)
		return label_names

	def run(self, dispatcher, tracker, domain):
		classes = ["adenomatous", "hyperplastic"]
		cnn_file = "keras_cnn_clf/polyp_cnn_clf_28000imgs_128hw.h5"
		patient_id = tracker.get_slot("patient_clf_id")
		if not patient_id:
			patient_id = 1
		print("Sending polyp classifier for Patient ID =", patient_id)

		f = h5py.File(cnn_file, mode="r")
		model_version = f.attrs.get("keras_version")
		keras_veresion = str(keras_version).encode("utf8")

		if model_version != keras_version:
			print("You are using Keras version ", keras_version,
				   ", but the model was built using ", model_version)

		model = load_model(cnn_file)

		base_img_path = "test/image/" + patient_id

		patient_polyp_img_filepaths, patient_polyp_img_filenames = self.get_filepaths(base_img_path)
		patient_polyp_images = self.get_images(patient_polyp_img_filepaths, 128)
		patient_polyp_img_nps = np.array(patient_polyp_images)
		patient_polyp_img_nps = patient_polyp_img_nps/255

		base_label_path = "test/Annotation/" + patient_id

		patient_polyp_label_filepaths, patient_polyp_label_filenames = self.get_filepaths(base_label_path)
		patient_polyp_labels = self.get_xml_label_names(patient_polyp_label_filepaths)
		label_enc = LabelEncoder()
		patient_polyp_labels = label_enc.fit_transform(patient_polyp_labels)

		# Run CNN Polyp Classification, Save Predicted Img, Utter Pred Image File
		y_pred = model.predict(patient_polyp_img_nps)
		y_pred_polyps = [0 if polyp < 0.5 else 1 for polyp in y_pred]

		p_id_frame_n = random.randint(0, len(patient_polyp_images)-1)
		plt.figure(figsize=(4,4))
		plt.title("Polyp Classification " + patient_id, fontsize=20)
		
		height = patient_polyp_img_nps[p_id_frame_n].shape[0]
		width = patient_polyp_img_nps[p_id_frame_n].shape[1]
		plt.text(width/10, height + 8.5,
			"Actual: {}\nPred: {}".format(
			classes[patient_polyp_labels[p_id_frame_n]], classes[y_pred_polyps[p_id_frame_n]]),
			va='top',
			fontsize=16)
		plt.axis("off")

		clf_polyp_img = plt.imshow(patient_polyp_img_nps[p_id_frame_n])
		save_dst="cnn_clf/trained_28000_aug/clf_pred_images"

		if not os.path.exists(save_dst):
			os.makedirs(save_dst)

		save_clf_polyp_file = "{}/pred_polyp_{}_{}.jpg".format(save_dst, classes[y_pred_polyps[p_id_frame_n]], p_id_frame_n)
		plt.savefig(save_clf_polyp_file, bbox_inches="tight")

		print("At Absolute Clf Image Path:", save_clf_polyp_file)
		dispatcher.utter_message(image=save_clf_polyp_file)
		# dispatcher.utter_message(text="Running Img Classifier")

		# reset slot
		return [SlotSet("patient_clf_id", None)]
	    

class ActionRunObjectDetection(Action):
	def name(self):
		return "action_run_object_detection"

	def get_filepaths(self, basepath, remove_ext=False):
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

	def read_img(self, img_path, flag = cv2.IMREAD_COLOR):
		# (height, width, 3)
		image = cv2.imread(img_path, flag)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image

	def get_images(self, image_group, size, flag = cv2.IMREAD_COLOR):
		images = []
		for image_path in image_group:
			image = self.read_img(image_path, flag)
			resized_img = cv2.resize(image, (size, size))
			images.append(resized_img)
		return images

	def get_xml_label_names(self, xml_files):
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

	def cv_yolo_detect(self, patient_id):
		# Hyperplastic Polyp Image 19 from video 3
		ADENOMATOUS = 0
		HYPERPLASTIC = 1
		classes = ["adenomatous", "hyperplastic"]

		# Read Darknet DNN M
		configPath = "yolov4/cfg/yolov4-custom.cfg"
		weightsPath = "yolov4/darknet/polyp_dtr/training/yolov4-custom_best.weights"
		yolo_net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
		layer_names = yolo_net.getLayerNames()
		# updating layer_names to be just unconnected output layers
		layer_names = [layer_names[layer_i-1] for layer_i in yolo_net.getUnconnectedOutLayers()]

		# Load Labels from obj.names
		labelsPath = "yolov4/data/obj.names"
		POLYP_LABELS = open(labelsPath).read().strip().split("\n")

		# Preprocessing Input Images: get_filepaths picks up both jpg and xml
		base_img_path = "test/image/" + str(patient_id)

		patient_polyp_img_filepaths, patient_polyp_img_filenames = self.get_filepaths(base_img_path)
		patient_polyp_images = self.get_images(patient_polyp_img_filepaths, 416)
		# randomly select one polyp frame from patient ID's video set
		p_id_frame_n = random.randint(0, len(patient_polyp_images)-1)
		# p_id_frame_n = 19 # first hyperplastic polyp detection success
		print("p_id_frame_n =", p_id_frame_n)

		base_label_path = "test/Annotation/" + str(patient_id)

		patient_polyp_label_filepaths, patient_polyp_label_filenames = self.get_filepaths(base_label_path)
		patient_polyp_labels = self.get_xml_label_names(patient_polyp_label_filepaths)
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
		conf_threshold = 0.15 # Conf Score threshold
		nms_threshold = 0.14 # IOU threshold

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

		# image = Image.fromarray(patient_polyp_images[p_id_frame_n].astype(np.uint8))

		# image.show()
		# decreasing size of image to make it fit better in Unity chat window
		patient_polyp_images[p_id_frame_n] = cv2.resize(patient_polyp_images[p_id_frame_n], (int(416*0.75), int(416*0.75)))
		patient_polyp_images[p_id_frame_n] = cv2.cvtColor(patient_polyp_images[p_id_frame_n], cv2.COLOR_BGR2RGB)

		save_dst="cv_yolov4/trained_7000_orig/dtc_images"
		apc_hpc_count = "apc_" + str(apc) + "_hpc_" + str(hpc)

		if not os.path.exists(save_dst):
			os.makedirs(save_dst)

		save_dtc_polyp_file = "{}/dtc_polyp_{}_{}.jpg".format(save_dst, apc_hpc_count, p_id_frame_n)
		cv2.imwrite(save_dtc_polyp_file, patient_polyp_images[p_id_frame_n])

		return save_dtc_polyp_file

	def run(self, dispatcher, tracker, domain):
		patient_id = tracker.get_slot("patient_dtr_id")
		if not patient_id:
			patient_id = 1
		print("Sending polyp image detection for Patient ID =", patient_id)

		out_dtc_polyp_file = self.cv_yolo_detect(patient_id)

		dispatcher.utter_message(image=out_dtc_polyp_file)
		# dispatcher.utter_message(text="Running Object Detection")

		# reset slot
		return [SlotSet("patient_dtr_id", None)]
	    
class ActionRunVideoDetection(Action):
	def name(self):
		return "action_run_video_detection"

	def run(self, dispatcher, tracker, domain):
		patient_id = tracker.get_slot("patient_dtr_id")
		print("Sending polyp video detection for Patient ID =", patient_id)

		# config = ConfigProto()
		# config.gpu_options.allow_growth = True
		# session = InteractiveSession(config=config)
		# STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
		# input_size = FLAGS.size
		# video_path = FLAGS.video + patient_id
		# polyp_video_path = os.listdir(video_path)

		# saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
		# infer = saved_model_loaded.signatures['serving_default']

		# # begin video capture
		# try:
		# 	vid = cv2.VideoCapture(int(polyp_video_path))
		# except:
		# 	vid = cv2.VideoCapture(polyp_video_path)

		# out = None

		# if FLAGS.output:
		# 	# by default VideoCapture returns float instead of int
		# 	width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
		# 	height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
		# 	fps = int(vid.get(cv2.CAP_PROP_FPS))
		# 	codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
		# 	out_dtr_video_file = FLAGS.output + patient_id + "/polyp_dtr_video_" + patient_id + ".mp4"
		# 	out = cv2.VideoWriter(out_dtr_video_file, codec, fps, (width, height))

		# while True:
		# 	return_value, frame = vid.read()
		# 	if return_value:
		# 		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# 		image = Image.fromarray(frame)
		# 	else:
		# 		print('Video has ended or failed, try a different video format!')
		# 		break
		
		# 	frame_size = frame.shape[:2]
		# 	image_data = cv2.resize(frame, (input_size, input_size))
		# 	image_data = image_data / 255.
		# 	image_data = image_data[np.newaxis, ...].astype(np.float32)
		# 	start_time = time.time()

		# 	batch_data = tf.constant(image_data)
		# 	pred_bbox = infer(batch_data)
		# 	for key, value in pred_bbox.items():
		# 		boxes = value[:, :, 0:4]
		# 		pred_conf = value[:, :, 4:]

		# 	boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
		# 		boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
		# 		scores=tf.reshape(
		# 			pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
		# 		max_output_size_per_class=50,
		# 		max_total_size=50,
		# 		iou_threshold=FLAGS.iou,
		# 		score_threshold=FLAGS.score
		# 	)
		# 	pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
		# 	image = utils.draw_bbox(frame, pred_bbox)
		# 	fps = 1.0 / (time.time() - start_time)
		# 	print("FPS: %.2f" % fps)
		# 	result = np.asarray(image)
		# 	# cv2.namedWindow("polyp_dtr_video_" + patient_id + ".mp4", cv2.WINDOW_AUTOSIZE)
		# 	result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			
		# 	# if not FLAGS.dont_show:
		# 	# 	cv2.imshow("result", result)
			
		# 	if FLAGS.output:
		# 		out.write(result)
		# 	if cv2.waitKey(1) & 0xFF == ord('q'): break
		# cv2.destroyAllWindows()

		# If using rasa-webchat, we can send video by URL. Wondering if can be done with .mp4 file too.
		# https://github.com/botfront/rasa-webchat

		# dispatcher.utter_message(attachment={
		# 	"type": "video",
		# 	"payload": {
		# 		"title": "Polyp Detection Video P_ID: " + patient_id,
		# 		"src": out_dtr_video_file
		# 	}
		# })
		dispatcher.utter_message(text="Running Video Detector")
		return []
	   
class ActionRunImageSegmentation(Action):
	def name(self):
		return "action_run_image_segmentation"
	
	def run(self, dispatcher, tracker, domain):
		dispatcher.utter_message(text="Running Image Segmentation")
		#in the place of text, return image segmentation model output
		return []
	
	
