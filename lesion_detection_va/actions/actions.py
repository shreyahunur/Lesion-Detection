# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

# NOTE: cd into actions folder, then run "rasa run actions", so
# it can pick up custom modules in core directory

# This is a simple example for a custom action which utters "Hello World!"
import os
# import cv2
import time
import numpy as np

# from typing import Any, Text, Dict, List
from rasa_sdk import Action #, Tracker
# from rasa_sdk.executor import CollectingDispatcher

# import matplotlib.pyplot as plt

# Keras CNN Polyp Classifer
# from tensorflow.keras.models import load_model
# import h5py
# from tensorflow.keras import __version__ as keras_version

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

	def run(self, dispatcher, tracker, domain):
		# classes = ["adenomatous", "hyperplastic"]
		# cnn_file = ".\keras_cnn_clf\jays_polyp_cnn_clf.h5"
		patient_id = tracker.get_slot("patient_clf_id")
		print("Sending polyp classifier for Patient ID =", patient_id)

		# f = h5py.File(cnn_file, mode="r")
		# model_version = f.attrs.get("keras_version")
		# keras_veresion = str(keras_version).encode("utf8")

		# if model_version != keras_version:
		# 	print("You are using Keras version ", keras_version,
		# 		   ", but the model was built using ", model_version)

		# model = load_model(cnn_file)

		# basepath = "test/image/" + patient_id
		# patient_polyp_img_filenames = os.listdir(basepath)
		# patient_polyp_img_file = os.path.join(basepath, patient_polyp_img_filenames[0])
		# patient_polyp_img = cv2.imread(patient_polyp_img_file)
		# patient_polyp_img = cv2.resize(patient_polyp_img, (120, 120))
		# patient_polyp_np = np.array(patient_polyp_img)
		# patient_polyp_np = patient_polyp_np/255

		# y_pred = model.predict(patient_polyp_np)
		# y_pred_polyps = [np.argmax(polyp) for polyp in y_pred]

		# # center text on image
		# font = cv2.FONT_HERSHEY_SIMPLEX
		# polyp_pred_label = "%s: %.2f" % (classes[y_pred_polyps[0]], y_pred)
		# textsize = cv2.getTextSize(polyp_pred_label, font, 1, 2)[0]
		# textX = (patient_polyp_img.shape[1] - textsize[0])/2
		# textY = (patient_polyp_img.shape[0] - textsize[1])/2
		# cv2.putText(patient_polyp_img, polyp_pred_label, (textX, textY), font, 1, (255,255,255), 2)
		# out_clf_img_file = "test/clf_img/" + patient_id + "/clf_img_" + patient_polyp_img_filenames[0] + ".jpg"
		# cv2.imwrite(out_clf_img_file, patient_polyp_img)

		# dispatcher.utter_message(image=out_clf_img_file)
		dispatcher.utter_message(text="Running Img Classifier")
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
	
	
