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

# from typing import Any, Text, Dict, List
from rasa_sdk import Action #, Tracker
from rasa_sdk.executor import CollectingDispatcher

import matplotlib.pyplot as plt

# Keras CNN Polyp Classifer
from tensorflow.keras.models import load_model
import h5py
from tensorflow.keras import __version__ as keras_version

import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder

# import sys

# sys.path.append("core")

# TensorFlow YoloV4 Polyp Detector
# import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# from absl import app, flags, logging
# from absl.flags import FLAGS

# import core.utils as utils
# from core.config import cfg
# from core.yolov4 import filter_boxes
# from tensorflow.python.saved_model import tag_constants
# from PIL import Image
# import cv2
# import numpy as np
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
# flags.DEFINE_string('weights', './checkpoints/yolov4-416',
#                     'path to weights file')
# flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
# flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
# flags.DEFINE_list('images', './test/image/', 'path to input image')
# flags.DEFINE_string('video', './test/video/', 'path to input video mp4')
# flags.DEFINE_string('output', "test/dtr_video/", 'path to output video')
# flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
# flags.DEFINE_float('iou', 0.45, 'iou threshold')
# flags.DEFINE_float('score', 0.25, 'score threshold')
# flags.DEFINE_boolean('dont_show', False, 'dont show image output')

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

	def run(self, dispatcher, tracker, domain):
		classes = ["adenomatous", "hyperplastic"]
		cnn_file = "keras_cnn_clf/polyp_cnn_clf_28000imgs_128hw.h5"
		patient_id = tracker.get_slot("patient_clf_id")
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
		patient_polyp_labels = self.get_txt_label_names(patient_polyp_label_filepaths)
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

		save_clf_polyp_file = "{}/pred_polyp_{}_{}.jpg".format(save_dst, classes[y_pred_polyps[p_id_frame_n]], 0)
		plt.savefig(save_clf_polyp_file, bbox_inches="tight")

		print("At Absolute Clf Image Path:", save_clf_polyp_file)
		dispatcher.utter_message(image=save_clf_polyp_file)
		# dispatcher.utter_message(text="Running Img Classifier")
		return []
	    

class ActionRunObjectDetection(Action):
	def name(self):
		return "action_run_object_detection"

	def run(self, dispatcher, tracker, domain):
		patient_id = tracker.get_slot("patient_dtr_id")
		print("Sending polyp image detection for Patient ID =", patient_id)

		# based on TF YoloV4 Detect.py code
		# config = ConfigProto()
		# config.gpu_options.allow_growth = True
		# session = InteractiveSession(config=config)
		# STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
		# input_size = FLAGS.size
		# images_path = FLAGS.images + patient_id
		# polyp_img_files = os.listdir(images_path)

		# # load model
		# saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])

		# # loop through images in list and run Yolov4 model on each
		# for count, image_path in enumerate(polyp_img_files, 1):
		# 	original_image = cv2.imread(image_path)
		# 	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

		# 	image_data = cv2.resize(original_image, (input_size, input_size))
		# 	image_data = image_data / 255.

		# 	images_data = []
		# 	for i in range(1):
		# 		images_data.append(image_data)
		# 	images_data = np.asarray(images_data).astype(np.float32)

		# 	infer = saved_model_loaded.signatures['serving_default']
		# 	batch_data = tf.constant(images_data)
		# 	pred_bbox = infer(batch_data)
		# 	for key, value in pred_bbox.items():
		# 		boxes = value[:, :, 0:4]
		# 		pred_conf = value[:, :, 4:]

		# 	print("1st boxes = ", boxes)
		# 	print("1st pred_conf = ", pred_conf)

		# 	boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
		# 		boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
		# 		scores=tf.reshape(
		# 			pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
		# 		max_output_size_per_class=50,
		# 		max_total_size=50,
		# 		iou_threshold=FLAGS.iou,
		# 		score_threshold=FLAGS.score
		# 	)
		# 	print("2nd boxes = ", boxes)
		# 	print("scores = ", scores)
		# 	print("classes = ", classes)
		# 	print("valid_detections = ", valid_detections)
		# 	pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

		# 	# read in all class names from config
		# 	# class_names = utils.read_class_names(cfg.YOLO.CLASSES)

		# 	# by default allow all classes in .names file
		# 	allowed_classes = ["adenomatous", "hyperplastic"]
			
		# 	# custom allowed classes (uncomment line below to allow detections for only people)
		# 	#allowed_classes = ['person']

		# 	image = utils.draw_bbox(original_image, pred_bbox, allowed_classes = allowed_classes)

		# 	image = Image.fromarray(image.astype(np.uint8))
		# 	# if not FLAGS.dont_show:
		# 	# 	image.show()
		# 	image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
		# 	out_dtr_img_file = "test/dtr_img/" + patient_id + "/dtr_img_" + image_path + ".jpg"
		# 	cv2.imwrite(out_dtr_img_file, image)

		# dispatcher.utter_message(image=out_dtr_img_file)
		dispatcher.utter_message(text="Running Object Detection")
		return []
	    
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
	
	
