# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
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
	def name(self) -> Text:
	    return "action_run_cnn"
		
	def run(self, dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

	    dispatcher.utter_message(text="Running Polyp Classifier")
	    return []
	
class ActionRunImageClassification(Action):
	def name(self) -> Text:
	    return "action_run_image_classification"
		
	def run(self, dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

	    dispatcher.utter_message(text="Running Image Classifier")
	    return []
	    
class ActionRunObjectDetection(Action):
	def name(self) -> Text:
	    return "action_run_object_detection"
		
	def run(self, dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

	    dispatcher.utter_message(text="Running Object Detector")
	    return []
	    
class ActionRunVideoDetection(Action):
	def name(self) -> Text:
	    return "action_run_video_detection"
		
	def run(self, dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

	    dispatcher.utter_message(text="Running Video Detector")
	    return []
	   
class ActionRunImageSegmentation(Action):
	def name(self) -> Text:
	    return "action_run_image_segmentation"
		
	def run(self, dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

	    dispatcher.utter_message(text="Running Image Segmentation")
	    #in the place of text, return image segmentation model output
	    return []
	
	
