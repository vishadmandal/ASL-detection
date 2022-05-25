from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import imutils
import time
import cv2
import numpy as np
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import tensorflow as tf
from matplotlib import pyplot as plt



outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

vs = VideoStream(src=1).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

	
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file('data/ASL_model/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('data/ASL_model/', 'ckpt-6')).expect_partial()

time.sleep(35)




def cam(frameCount):
	# # Load pipeline config and build a detection model
	# configs = config_util.get_configs_from_pipeline_file('data/ASL_model/pipeline.config')
	# detection_model = model_builder.build(model_config=configs['model'], is_training=False)

	# # Restore checkpoint
	# ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
	# ckpt.restore(os.path.join('data/ASL_model/', 'ckpt-6')).expect_partial()
	global vs, outputFrame, lock
	total = 0
	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)


	#--------------------------------------------------#
		category_index = label_map_util.create_category_index_from_labelmap('data/annotations/letter_map.pbtxt')
		image_np = np.array(frame)
		input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
		detections = detect_fn(input_tensor)
		
		num_detections = int(detections.pop('num_detections'))
		detections = {key: value[0, :num_detections].numpy()
					for key, value in detections.items()}
		detections['num_detections'] = num_detections

		detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
		label_id_offset = 1
		frame = image_np.copy()
		viz_utils.visualize_boxes_and_labels_on_image_array(
					frame,
					detections['detection_boxes'],
					detections['detection_classes']+label_id_offset,
					detections['detection_scores'],
					category_index,
					use_normalized_coordinates=True,
					max_boxes_to_draw=5,
					min_score_thresh=.5,
					agnostic_mode=False)

	#--------------------------------------------------#


		total += 1
		with lock:
			outputFrame = frame







@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections




def generate():
	global outputFrame, lock

	while True:

		with lock:
			if outputFrame is None:
				continue
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			if not flag:
				continue
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	

	t = threading.Thread(target=cam, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
vs.stop()
