import json
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from yolact import video
import cv2
from time import sleep
from vidgear.gears import WriteGear
import itertools
from math import sqrt
import random
from cfg import cfg

from trackers.tracker_centroid import CentroidTracker
from utils import (
	parabolic_func, get_bbox_params, get_distance_btw_dots,
	average, get_rotated_bounding_rectangle, get_rect_rotation,
	upscale_poly_cap, upscale_bbox, check_bbox_coordinates,
	get_centroid, get_bbox_coords, read_json, get_iou,
	get_area_cap, get_side_cap
)
from utils_video import get_stream_params, mmcv_process_image, init_mmcv_model
import numpy as np

trackableObjects = {}


def find_cam_quality(
		camera_num: int,
		vid: str,
		temp_camera_dir: str,
		start_second: int = 0,
		camera_quality_img_path: str = None,
		verbose: bool = False,
		gpu_id: int = None
):
	"""
	1. Save to the temp video file first 60 seconds of the input video.
	2. Find ballot boxes and lids.
	3. Classify ballot boxes. Pretrained weights support following types:
		*	"ballot_box" - regular ballot box;
		* 	"koib" - electronic ballot box.

	Args:
		camera_num					- camera number on the polling station
		vid							- path to the source video
		temp_camera_dir				- directory path with temp files
		start_second				- starting second of processing video. In case if video
		 								starts earlier than opening hour of polling station.
		camera_quality_img_path		- custom path for saving image with visualized boxes.
										By default, images will be saved to temp dir.
		verbose						- print processing docs
		gpu_id						- which GPU to utilize

	Returns:
		1) boxes_bool				- whether ballot boxes are visible on the video
		2) obj_num					- number of ballot boxes on the video
		3) objects_dict				- ballot boxes docs

		output_true_objects[obj_id] = {			- object_id os a key. Each value consists of:
			'bbox': {
				'x1': ..., 'y1':  ..., 			- top left (x1, y1) and bottom right (x2, y2) bbox coordinates
				'y2': ..., 'y2': ...
			},
			'caps': {
				'type': ...,					- ballot box lid / cap type
												('bbox' - orthogonal rectangle, 'poly' - rotated rectangle)
				'ort_bbox': {					- orthogonal rectangle coordinates
					'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...
				},
				'ort_bbox_k':{                  - normalized orthogonal rectangle coordinates (0...1)
					'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...
				}
				'rot_bbox': {					- rotated rectangle coordinates
					'top_right': {'x': ..., 'y': ...},
					'top_left': {'x': ..., 'y': ...},
					'bottom_left': {'x': ..., 'y': ...},
					'bottom_right': {'x': ..., 'y': ...},
				},
				'rot_bbox_k': {					- normalized rotated rectangle coordinates
					'top_right': {'x': ..., 'y': ...},
					'top_left': {'x': ..., 'y': ...},
					'bottom_left': {'x': ..., 'y': ...},
					'bottom_right': {'x': ..., 'y': ...},
				},
				'upscaled_rot_bbox': {			- upscaled rotated rectangle coordinates
					'top_right': {'x': ..., 'y': ...},
					'top_left': {'x': ..., 'y': ...},
					'bottom_left': {'x': ..., 'y': ...},
					'bottom_right': {'x': ..., 'y': ...}
				},
				'upscaled_rot_bbox_k': {			- upscaled normalized rotated rectangle coordinates
					'top_right': {'x': ..., 'y': ...},
					'top_left': {'x': ..., 'y': ...},
					'bottom_left': {'x': ..., 'y': ...},
					'bottom_right': {'x': ..., 'y': ...}
				},
				'upscaled_ort_bbox': {				- upscaled rotated rectangle coordinates
					'x1': ..., 'y1': ...,
					'x2': ..., 'y2': ...
				},
				'upscaled_ort_bbox_k': {			- upscaled normalized rotated rectangle coordinates
					'x1': ..., 'y1': ...,
					'x2': ..., 'y2': ...
				},
				'rect_angle': ...,				- rotation angle of rectangle
				'centroid_k': {'x': ..., 'y': ...},		- normalized lid centroid coordinates
				'width_k': ...,					- normalized lid's width
				'height_k': ...,				- normalized lid's height
				'mask_points': [ [x, y], ... ],	- list of mask points
				'area': ...,                    - area of the lid
				'side_bool': True/False,        - whether lid is close to the of the frame sides
				'side': None/[...]              - name of the closest sides (if there are any)
			},
			'centroid_k': {
				'x':  ..., 'y': ...				- normalized coordinates of ballot box centroid
			},
			'obj_quality': 						- "quality" param of the ballot box
			'conf': 							- yolact confidence in the ballot box prediction
			'type': 							- ballot box type
													('koib' - electronic, 'ballot_box' - regular)
			'type_conf':						- QueryInst confidence in the type prediction
			'normalized_orientation_k':			- normalized orientation of the box
			'obj_params': {						- parameters used for the ballot box "quality" computation
				'distance_from_center_k': 		- distance from the frame's center (0 ... 0.6)
				'width_k': 						- ratio of ballot box width to the frame width
				'height_k': 					- ratio of ballot box height to the frame height
				'area_k': 						- box area coefficient (fraction of the total frame area)
				'visible_rate': 				- frames fraction when the ballot box is visible
				'intersection_rate': 			- frames fraction when the ballot box isn't intersected by smth or someone
				'normalized_dist_k':			- normalized distance from the frame's center (0 ... 1)
				'normalized_width_k':			- normalized ratio of the ballot box width to the frame width (0...1)
			}
		}

		4) obj_images				- source frame and cropped ballot boxes images (each img is a np array)

		output_true_images = {
			'overall_plan': {					- best overall plan images
				'source_frame':					- resized source frame
				'recognized_frame':				- frame with visualized bboxes and masks
				'processed_frame': 				- frame with visualized ballot boxes docs
			},
			'objects': {						- each obj_id represents a unique ballot box
				obj_id: {
					'source_frame':				- cropped source ballot box image
					'recognized_frame':			- cropped image with visualized data
				},
				...
			}
		}

		5) cam_quality							- quality parameter of the camera (0...1)
	"""
	global trackableObjects
	trackableObjects = {}

	if not os.path.exists(temp_camera_dir):
		os.makedirs(temp_camera_dir)

	short_vid_boxes_recs = os.path.join(temp_camera_dir, '{}_recs.json'.format(camera_num))

	# Read first frame of the video
	video_stream = cv2.VideoCapture(vid)
	got_frame, source_frame = video_stream.read()
	if not got_frame:
		while not got_frame:
			got_frame, source_frame = video_stream.read()
	process = True if got_frame else False

	# Get fps of the video and parameters for resizing
	count_frames = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
	video_fps, source_height, source_width, resized_height, resized_width, aspect_ratio = \
		get_stream_params(video_stream, source_frame, cfg.target_pixels)

	if verbose:
		print("FPS: {}, W: {}, H: {}, resized_width: {}, resized_height: {}".format(
			video_fps, source_width, source_height, resized_width, resized_height
		))

	start_frame = int(start_second * video_fps)

	# We will get each analyzing_frame for saving to temp video
	analyzing_frames = int((count_frames - start_frame) / cfg.count_analyzed_frames)

	# Find number of frames to save in temp video
	# temp_video_frames_num = video_fps * cfg.find_boxes_video_length
	temp_video_frames_num = cfg.count_analyzed_frames
	temp_video_path = os.path.join(temp_camera_dir, '{}_temp_video.mp4'.format(camera_num))

	# cv2 VideoWriter will not work in our case - yolact's video.py
	# will raise an error
	output_params = {
		"-vcodec": "libx264", "-crf": 0,
		"-preset": "fast", '-input_framerate': cfg.target_fps
	}
	writer = WriteGear(
		output_filename=temp_video_path,
		compression_mode=True,
		logging=False,
		**output_params
	)

	# Save temp video for ballot boxes recognition
	if verbose:
		print('Saving {} frames of stream for finding boxes... '.format(
			cfg.count_analyzed_frames), end=''
		)

	if start_frame != 0:
		video_stream.set(1, start_frame)
	temp_frame_id = 1
	count_write_frames = 0
	while process:
		if temp_frame_id % analyzing_frames == 0:
			frame_resized = cv2.resize(source_frame, (resized_width, resized_height))
			writer.write(frame_resized)
			count_write_frames += 1
		temp_frame_id += 1
		if count_write_frames >= temp_video_frames_num or temp_frame_id >= count_frames:
			process = False
		else:
			got_frame, source_frame = video_stream.read()
			process = True if got_frame else False

	video_stream.release()
	writer.close()
	sleep(0.05)
	if verbose:
		print('Done.')

	# Find ballot boxes in temp video
	if verbose:
		print('Finding boxes...')

	out_path = os.path.join(
		temp_camera_dir, '{}_recognized_temp_video.mp4'.format(camera_num)
	) if cfg.save_rec_video else None

	# raw_recognitions contains following:
	# frame id (key):
	#   "frame_id": id,
	#   "frame_width": w,
	#   "frame_height": h,
	# 	"objects": {}        - if there is no boxes
	#   "objects":           - if boxes are found
	#      "class_name": class,
	#      "conf": score,
	#      "coords": [
	#         [x1, y1],
	#         [x2, y2]
	# ...
	raw_recognitions, raw_masks = video.process_video(
		cfg.models.boxes.config, cfg.models.boxes.weights,
		temp_video_path, cfg.parallel_frames, out_path, verbose,
		gpu_id=gpu_id
	)

	# Save json with recognitions docs to temp dir
	with open(short_vid_boxes_recs, "w") as j:
		json.dump(raw_recognitions, j, indent=2)

	raw_recognitions = read_json(short_vid_boxes_recs)
	if verbose:
		print('Done.')

	# Analyze raw recognitions:
	# 	* find "true" ("real") boxes - try to remove false positives
	# 	* compute ballot boxes parameters
	# 	* get ballot boxes images
	# 	* compute camera quality (how it is good for voter turnout counting)
	if verbose:
		print('Analyzing {} frames with recognitions... '.format(len(raw_recognitions)))

	boxes_bool, obj_num, objects_dict, obj_images, cam_quality = \
		analyze_short_video_recs(
			raw_recognitions, raw_masks, camera_num,
			len(raw_recognitions), temp_video_path, temp_camera_dir,
			resized_width, resized_height,
			cam_quality_img_path=camera_quality_img_path,
			gpu_id=gpu_id
		)

	if verbose:
		print('Found {} boxes! Camera quality {:.2f}%'.format(obj_num, cam_quality * 100))

	# Delete temp video if specified
	if cfg.delete_temp_video:
		os.remove(temp_video_path)

	if cfg.save_json_output:
		with open(os.path.join(temp_camera_dir, '{}_objects.json'.format(camera_num)), "w") as j:
			json.dump(objects_dict, j, indent=2)

	return boxes_bool, obj_num, objects_dict, obj_images, cam_quality


class TrackableObject:
	def __init__(self, objectID, centroid):
		self.objectID = objectID
		self.centroids = [centroid]

		self.matched_in_current_frame = False
		self.visible_frames = 0
		self.visible_rate = 0.0

		self.intersected_frames = 0
		self.intersection_rate = 0.0

		self.coords_list = []
		self.max_conf = 0.0
		self.max_conf_area = 0
		self.max_conf_frame_id = None
		self.true_coords = None


def normalize_bbox_width_k(x):
	"""
	Normalize the box width coefficient to 0 ... 1 value.
	The closer to 1, the bigger the box.
	"""
	if x < 0.03:
		return x / 2
	elif x > 0.24:
		return 1
	else:
		return 12.9739 * x * x * x - 19.9623 * x * x + 7.92655 * x + 0.0619943


def get_rec_threshold(x):
	"""
	Normalize the value of max confidence in the ballot box detection.
	"""

	if x <= 0.0:
		return 1
	if 0.0 < x <= 0.4:
		return 1 - x * x
	elif 0.4 < x <= 0.8:
		return 1 - (1.9 * x - 0.6)
	else:
		return 1 - (0.2 * x + 0.79)


def normalize_centroid_distance(x):
	"""
	Normalize the value of distance between box centroid and frame centroid.
	Return values from 0 to 1.

	The closer centroid is located to the center of the frame,
	the closer normalized value to 1.
	"""
	if x > 0.684931507:
		return 0
	else:
		return 1 - parabolic_func(x * 1.46)


def get_video_stream_params(stream, src_frame, target_pixels, find_fps=True):
	if find_fps:
		vid_fps = int(stream.get(cv2.CAP_PROP_FPS))
	(src_height, src_width) = src_frame.shape[:2]

	aspect_ratio = src_height / src_width
	resized_width = int(sqrt(target_pixels / aspect_ratio))
	resized_height = int(aspect_ratio * resized_width)

	return vid_fps if find_fps else None, src_height, src_width, \
		resized_height, resized_width, aspect_ratio


def get_matched_true_boxes_ids(true_coords, rec_coords):
	"""
	- Take "true" ballot boxes bboxes and boxes recognitions.
	- Define if the bboxes centroids match
	- Return matched bboxes
	"""
	matches = {}
	rec_conf = {}
	for true_obj_id in true_coords.keys():
		for obj_id in rec_coords.keys():
			c_x_bbox1, c_y_bbox1, _, __ = get_bbox_params(
				int(true_coords[true_obj_id][0][0]),
				int(true_coords[true_obj_id][0][1]),
				int(true_coords[true_obj_id][1][0]),
				int(true_coords[true_obj_id][1][1])
			)
			c_x_bbox2, c_y_bbox2, _, __ = get_bbox_params(
				int(rec_coords[obj_id]['coords'][0][0]),
				int(rec_coords[obj_id]['coords'][0][1]),
				int(rec_coords[obj_id]['coords'][1][0]),
				int(rec_coords[obj_id]['coords'][1][1])
			)

			rec_conf[obj_id] = rec_coords[obj_id]['conf']

			distance_to_true_centroid = get_distance_btw_dots(
				int(c_x_bbox1), int(c_y_bbox1), int(c_x_bbox2), int(c_y_bbox2)
			)

			if distance_to_true_centroid < 15:
				matches[true_obj_id] = (obj_id, rec_conf[obj_id])

	return matches


def get_true_frame(stream, frame_id, gpu_id=None):
	"""
	Recognize ballot boxes on one target frame_id
	"""
	stream.set(1, frame_id)
	got_fr, frame = stream.read()
	if got_fr:
		vid_fps, src_height, src_width, resized_height, resized_width, _ = \
			get_stream_params(stream, frame, cfg.target_pixels)
		frame_res = cv2.resize(frame, (resized_width, resized_height))
		rec_image, _, __ = video.process_image(
			cfg.models.boxes.config, cfg.models.boxes.weights, frame_res,
			gpu_id=gpu_id
		)
		return frame_res, rec_image
	else:
		return None, None


def get_objects_images(
		obj_coords, max_conf_frame_ids, vid_path, raw_recs, obj_num,
		gpu_id=None
):
	"""
	- Find the best frame where all ballot boxes are best seen
		(with the largest conf of all boxes)
	- Read the best frame from the temp video, recognize boxes and draw masks

	Returns:

	images = {
		'overall_plan': {
			'source': frame_res,					- Source frame (np array)
			'recognized': rec_image,				- Frame with visualized masks (np array)
			'frame_id': overall_plan_frame_id		- Frame_id of the best frame
		},
		'objects': {
			1: {									- ID of the "true" ballot box
				'frame_id': frame_id,
				'conf': conf,
				'source': frame_res,				- cropped frame with a ballot box
				'recognized': rec_image,			- cropped frame with a box + mask
				'bbox_coords': obj_coords           - bbox coordinates
			},
			...
		}
	}
	"""
	overall_plan_frame_id = None
	all_objects_in_one_frame = {}
	true_objects_frames_conf = {}
	max_conf_frames = {}

	for true_obj_id in obj_coords.keys():
		max_conf_frames[true_obj_id] = {'matched_frame_id': None, 'conf': None}
		true_objects_frames_conf[true_obj_id] = []

	# Define, if the found object is a "true" (real) ballot box
	for raw_rec in raw_recs:
		matched_objects = get_matched_true_boxes_ids(obj_coords, raw_rec['objects'])
		frame_id = raw_rec['frame_id']

		# Save ballot boxes bboxes to a separate dictionary
		if len(matched_objects) == obj_num:
			all_objects_in_one_frame[frame_id] = matched_objects

	# When frame contains multiple ballot boxes - find best
	# frame with maxed conf of each box
	if obj_num >= 2:
		if all_objects_in_one_frame:
			conf_sum = {}
			for true_obj_id in true_objects_frames_conf.keys():
				for frame_id in all_objects_in_one_frame.keys():
					confidence = all_objects_in_one_frame[frame_id][true_obj_id][1]
					true_objects_frames_conf[true_obj_id].append([frame_id, float(confidence)])

				# Find the sum of confidences
				for [fr_id, conf] in true_objects_frames_conf[true_obj_id]:
					if fr_id not in conf_sum.keys():
						conf_sum[fr_id] = conf
					else:
						conf_sum[fr_id] += conf

			# Get frame_id where sum of confidences is the largest
			overall_plan_frame_id = max(conf_sum, key=conf_sum.get)

	# When there is only one box in the frame - get frame with max conf
	elif obj_num == 1:
		random_true_obj_id = random.choice(list(max_conf_frame_ids.keys()))
		overall_plan_frame_id = max_conf_frame_ids[random_true_obj_id]['frame_id']

	# For each ballot box save frame ids with maxed conf
	for true_obj_id in max_conf_frames.keys():
		max_conf_frames[true_obj_id]['matched_frame_id'] = max_conf_frame_ids[true_obj_id]['frame_id']
		max_conf_frames[true_obj_id]['conf'] = max_conf_frame_ids[true_obj_id]['conf']

	vid_stream = cv2.VideoCapture(vid_path)

	if overall_plan_frame_id:
		frame_res, rec_image = get_true_frame(vid_stream, overall_plan_frame_id, gpu_id=gpu_id)

	else:
		random_true_obj_id = random.choice(list(max_conf_frame_ids.keys()))
		overall_plan_frame_id = max_conf_frames[random_true_obj_id]['matched_frame_id']
		frame_res, rec_image = get_true_frame(vid_stream, overall_plan_frame_id, gpu_id=gpu_id)

	images = {
		'overall_plan': {
			'source': frame_res,
			'recognized': rec_image,
			'frame_id': overall_plan_frame_id
		},
		'objects': {}
	}

	# Prepare output dict
	for true_obj_id in max_conf_frames.keys():
		frame_id = max_conf_frames[true_obj_id]['matched_frame_id']

		conf = max_conf_frames[true_obj_id]['conf']
		frame_res, rec_image = get_true_frame(vid_stream, frame_id, gpu_id=gpu_id)

		images['objects'][true_obj_id] = {
			'frame_id': frame_id,
			'conf': conf,
			'source': frame_res,
			'recognized': rec_image,
			'bbox_coords': obj_coords[true_obj_id]
		}

	vid_stream.release()

	return images


def recognize_objects_types(images, out_path=None, gpu_id=None):
	"""
	Classify ballot box type based on lid's segmentation results
	"""

	recognized_objects = {}
	mask = {}

	for true_obj_id in images['objects'].keys():
		recognized_objects[true_obj_id] = {
			'type': 'n/d',
			'type_conf': 0.0,
			'max_iou_cap': 0.0,
		}
		mask[true_obj_id] = {}

	# Initialize lids segmentation model
	model = init_mmcv_model(
		cfg.models.boxes_cap.config,
		cfg.models.boxes_cap.weights,
		gpu_id=gpu_id
	)

	# Segmentate ballot box lid, parse results and prepare output dict
	for true_obj_id in images['objects'].keys():
		image = images['objects'][true_obj_id]['source']
		rec_image, pre_rec_info, rec_masks = mmcv_process_image(model, image)

		rec_info = pre_rec_info[0]['objects'] if pre_rec_info else None

		for true_mask_id in images['objects'].keys():
			box_type = None
			conf = None
			bbox = images['objects'][true_mask_id]['bbox_coords']

			bbox_box = {
				'x1': int(bbox[0][0]), 'x2': int(bbox[1][0]),
				'y1': int(bbox[0][1]), 'y2': int(bbox[1][1]),
			}

			if rec_info:

				max_iou = 0
				for cap_id, check_bbox in rec_info.items():
					bbox_cap = {
						'x1': int(check_bbox['coords'][0][0]),
						'x2': int(check_bbox['coords'][1][0]),
						'y1': int(check_bbox['coords'][0][1]),
						'y2': int(check_bbox['coords'][1][1]),
					}
					iou, iou_box, iou_cap = get_iou(bbox_box, bbox_cap)
					if iou > max_iou and iou_cap > 0.65 and \
						iou_cap > recognized_objects[true_mask_id]['max_iou_cap']:

						max_iou = iou
						box_type = check_bbox['class_name']
						conf = check_bbox['conf']
						mask_true = rec_masks[0]['objects'][cap_id]
						bbox_true = check_bbox['coords']

				if box_type and recognized_objects[true_mask_id]['type_conf'] < float(conf):
					recognized_objects[true_mask_id] = {
						'type': box_type,
						'type_conf': float(conf),
						'max_iou_cap': iou_cap
					}
					mask[true_mask_id] = {
						'mask_points': mask_true,
						'bbox_x1': int(bbox_true[0][0]),
						'bbox_y1': int(bbox_true[0][1]),
						'bbox_x2': int(bbox_true[1][0]),
						'bbox_y2': int(bbox_true[1][1]),
						'c_x': int((int(bbox_true[1][0]) + int(bbox_true[0][0])) / 2),
						'c_y': int((int(bbox_true[1][1]) + int(bbox_true[0][1])) / 2),
						'bbox_width': int(bbox_true[1][0]) - int(bbox_true[0][0]),
						'bbox_height': int(bbox_true[1][1]) + int(bbox_true[0][1]),
					}

	return recognized_objects, mask


def analyze_cam_quality(images, objects_coords, obj_types, H, W, draw_boxes):
	"""
	Compute camera "quality" parameter - how camera suits
	for voter turnout counting
	"""

	global trackableObjects

	size_weight = 0.6
	position_weight = 0.4

	font_face = cv2.FONT_HERSHEY_DUPLEX
	font_scale = 0.6
	font_thickness = 1

	height = int(H)
	width = int(W)

	# Visualize bboxes of ballot boxes
	if draw_boxes and images['overall_plan']:
		frame = images['overall_plan']['source'].copy()
		for obj in objects_coords.keys():
			cv2.rectangle(
				frame,
				(int(objects_coords[obj][0][0]), int(objects_coords[obj][0][1])),
				(int(objects_coords[obj][1][0]), int(objects_coords[obj][1][1])),
				(21, 225, 255), 1
			)
	else:
		frame = None

	# Compute quality factors for each ballot box
	factors = {}
	for object_id in objects_coords.keys():
		to = trackableObjects.get(object_id, None)

		# Find coefficients of centroid, width and height
		obj_centroid_x_k, obj_centroid_y_k, obj_width_k, obj_height_k = \
			get_bbox_params(
				objects_coords[object_id][0][0] / width,
				objects_coords[object_id][0][1] / height,
				objects_coords[object_id][1][0] / width,
				objects_coords[object_id][1][1] / height
			)

		# Ballot box area coefficient
		obj_area_k = obj_width_k * obj_height_k

		# Find distance between middle of the frame and
		# ballot box centroid
		distance_from_center = get_distance_btw_dots(
			obj_centroid_x_k, obj_centroid_y_k, 0.5, 0.5
		)

		factors[object_id] = {
			'distance_from_center': distance_from_center,
			'width_k': obj_width_k,
			'height_k': obj_height_k,
			'area_k': obj_area_k,
			'centroid_k': [obj_centroid_x_k, obj_centroid_y_k],
			'visible_rate': to.visible_rate,
			'intersection_rate': to.intersection_rate,
			'conf': to.max_conf
		}

	objects_quality = []
	text_info = []
	for obj_id in factors.keys():
		(centroid_x_k, centroid_y_k) = factors[obj_id]['centroid_k']
		dist_from_center = factors[obj_id]['distance_from_center']
		width_k = factors[obj_id]['width_k']

		# not used:
		# height_k = factors[obj_id]['height_k']
		# area_k = factors[obj_id]['area_k']
		# visible_rate = factors[obj_id]['visible_rate']
		# intersection_rate = factors[obj_id]['intersection_rate']
		# conf = factors[obj_id]['conf']

		# Normalize distance parameter
		normalized_dist_k = normalize_centroid_distance(dist_from_center)

		# Normalize width parameter
		normalized_width_k = normalize_bbox_width_k(width_k)

		object_quality = (
			normalized_width_k * size_weight +
			normalized_dist_k * position_weight * normalized_width_k
		)
		objects_quality.append(object_quality)

		factors[obj_id]['normalized_dist_k'] = normalized_dist_k
		factors[obj_id]['normalized_width_k'] = normalized_width_k
		factors[obj_id]['object_quality'] = object_quality

		if draw_boxes:

			cv2.circle(
				frame,
				(int(centroid_x_k * width), int(centroid_y_k * height)),
				4, (199, 248, 255), -1
			)

			obj_type = obj_types[obj_id]['type'] \
				if obj_types[obj_id]['type'] else 'n/d'
			obj_type_conf = obj_types[obj_id]['type_conf'] * 100 \
				if obj_types[obj_id]['type_conf'] else 0.0

			cv2.putText(
				frame,
				'{}, {}'.format(obj_id, obj_type),
				(int(centroid_x_k * width) + 5, int(centroid_y_k * height) - 10),
				font_face, font_scale, (0, 0, 255), font_thickness
			)
			text_info.append((
				'ID {} | Quality: {:.2f}%, distance: {:.2f}%, '
				'width: {:.2f}% | Type: {}, conf {:.2f}%'.format(
					obj_id,
					object_quality * 100,
					normalized_dist_k * 100,
					normalized_width_k * 100,
					obj_type,
					obj_type_conf
				)
			))

	# Compute camera quality parameter as a average
	# of boxes quality values
	cam_quality = average(objects_quality)

	if draw_boxes:
		text_info.append(('Camera quality: {:.2f}%'.format(cam_quality * 100)))
		for i, k in enumerate(text_info):
			text = "{}".format(k)
			cv2.putText(
				frame, text,
				(10, height - ((i * 20) + 20)),
				font_face, 0.4, (212, 255, 255), 1
			)

	return frame, cam_quality, factors


def get_true_object_coords(vid_frames_total, frame):
	"""
	Filter duplicating and false positive ballot boxes
	"""
	global trackableObjects

	true_obj_ids = []
	true_objects_coords = {}
	max_conf_frame_ids = {}

	# Find final coordinates of ballot boxes and remove intersected boxes
	objects_to_check = {'objects': {}}
	for objectID in trackableObjects.keys():
		to = trackableObjects.get(objectID, None)

		x1_list, y1_list, x2_list, y2_list = [], [], [], []
		for coords in to.coords_list:
			x1, y1, x2, y2 = coords[0][0], coords[0][1], coords[1][0], coords[1][1]
			x1_list.append(x1)
			y1_list.append(y1)
			x2_list.append(x2)
			y2_list.append(y2)

		true_x1, true_y1, true_x2, true_y2 = \
			int(average(x1_list)), int(average(y1_list)), \
			int(average(x2_list)), int(average(y2_list))

		to.true_coords = {'x1': true_x1, 'y1': true_y1, 'x2': true_x2, 'y2': true_y2}

		objects_to_check['objects'][objectID] = {}
		objects_to_check['objects'][objectID]['coords'] = [
			[to.true_coords['x1'], to.true_coords['y1']],
			[to.true_coords['x2'], to.true_coords['y2']]
		]

	output_objects = intersection_check(objects_to_check, cfg.intersection_threshold)

	for objectID in output_objects['objects'].keys():
		to = trackableObjects.get(objectID, None)

		# Find threshold value of frames to assume that box is "true"
		recognized_frames_threshold = int(
			get_rec_threshold(to.max_conf) * vid_frames_total
		)

		if to.visible_frames > recognized_frames_threshold:
			true_obj_ids.append(objectID)

			true_objects_coords[objectID] = [
				[to.true_coords['x1'], to.true_coords['y1']],
				[to.true_coords['x2'], to.true_coords['y2']]
			]
			max_conf_frame_ids[objectID] = {
				'conf': to.max_conf,
				'frame_id': to.max_conf_frame_id
			}

		trackableObjects[objectID] = to

	true_objects_num = len(true_obj_ids)

	return true_objects_coords, true_objects_num, max_conf_frame_ids


def make_output_caps_json(
		output_caps, true_obj_id, cap_type, w, h, side_cap,
		area=None,
		mask=None,
		cap_coords=None,
		rect_points=None,
		rotated_poly_upscaled=None,
		rect_angle=None,
		upscaled_cap_coords=None,
		cap_centroids=None,
		cap_params=None,
		upscaled_ort_bbox=None
):
	"""
	Construct dictionary with ballot box's lid docs
	"""
	output_caps[true_obj_id] = {
		'type': 'poly' if cap_type == 'poly' else 'bbox',
		'ort_bbox': {
			'x1': mask['bbox_x1'] if cap_type == 'poly' else cap_coords[true_obj_id]['x1'],
			'y1': mask['bbox_y1'] if cap_type == 'poly' else cap_coords[true_obj_id]['y1'],
			'x2': mask['bbox_x2'] if cap_type == 'poly' else cap_coords[true_obj_id]['x2'],
			'y2': mask['bbox_y2'] if cap_type == 'poly' else cap_coords[true_obj_id]['y2']
		},
		'ort_bbox_k': {
			'x1': mask['bbox_x1'] / w if cap_type == 'poly' else cap_coords[true_obj_id]['x1'] / w,
			'y1': mask['bbox_y1'] / h if cap_type == 'poly' else cap_coords[true_obj_id]['y1'] / h,
			'x2': mask['bbox_x2'] / w if cap_type == 'poly' else cap_coords[true_obj_id]['x2'] / w,
			'y2': mask['bbox_y2'] / h if cap_type == 'poly' else cap_coords[true_obj_id]['y2'] / h
		},
		'rot_bbox': {
			'top_right': {
				'x': rect_points[0][0] if cap_type == 'poly' else None,
				'y': rect_points[0][1] if cap_type == 'poly' else None
			},
			'top_left': {
				'x': rect_points[1][0] if cap_type == 'poly' else None,
				'y': rect_points[1][1] if cap_type == 'poly' else None
			},
			'bottom_left': {
				'x': rect_points[2][0] if cap_type == 'poly' else None,
				'y': rect_points[2][1] if cap_type == 'poly' else None
			},
			'bottom_right': {
				'x': rect_points[3][0] if cap_type == 'poly' else None,
				'y': rect_points[3][1] if cap_type == 'poly' else None
			}
		},
		'rot_bbox_k': {
			'top_right': {
				'x': rect_points[0][0] / w if cap_type == 'poly' else None,
				'y': rect_points[0][1] / h if cap_type == 'poly' else None
			},
			'top_left': {
				'x': rect_points[1][0] / w if cap_type == 'poly' else None,
				'y': rect_points[1][1] / h if cap_type == 'poly' else None
			},
			'bottom_left': {
				'x': rect_points[2][0] / w if cap_type == 'poly' else None,
				'y': rect_points[2][1] / h if cap_type == 'poly' else None
			},
			'bottom_right': {
				'x': rect_points[3][0] / w if cap_type == 'poly' else None,
				'y': rect_points[3][1] / h if cap_type == 'poly' else None
			}
		},
		'upscaled_rot_bbox': {
			'top_right': {
				'x': rotated_poly_upscaled['top_right']['x'] if cap_type == 'poly' else None,
				'y': rotated_poly_upscaled['top_right']['y'] if cap_type == 'poly' else None
			},
			'top_left': {
				'x': rotated_poly_upscaled['top_left']['x'] if cap_type == 'poly' else None,
				'y': rotated_poly_upscaled['top_left']['y'] if cap_type == 'poly' else None
			},
			'bottom_left': {
				'x': rotated_poly_upscaled['bottom_left']['x'] if cap_type == 'poly' else None,
				'y': rotated_poly_upscaled['bottom_left']['y'] if cap_type == 'poly' else None
			},
			'bottom_right': {
				'x': rotated_poly_upscaled['bottom_right']['x'] if cap_type == 'poly' else None,
				'y': rotated_poly_upscaled['bottom_right']['y'] if cap_type == 'poly' else None
			}
		},
		'upscaled_rot_bbox_k': {
			'top_right': {
				'x': rotated_poly_upscaled['top_right']['x'] / w if cap_type == 'poly' else None,
				'y': rotated_poly_upscaled['top_right']['y'] / h if cap_type == 'poly' else None
			},
			'top_left': {
				'x': rotated_poly_upscaled['top_left']['x'] / w if cap_type == 'poly' else None,
				'y': rotated_poly_upscaled['top_left']['y'] / h if cap_type == 'poly' else None
			},
			'bottom_left': {
				'x': rotated_poly_upscaled['bottom_left']['x'] / w if cap_type == 'poly' else None,
				'y': rotated_poly_upscaled['bottom_left']['y'] / h if cap_type == 'poly' else None
			},
			'bottom_right': {
				'x': rotated_poly_upscaled['bottom_right']['x'] / w if cap_type == 'poly' else None,
				'y': rotated_poly_upscaled['bottom_right']['y'] / h if cap_type == 'poly' else None
			}
		},
		'upscaled_ort_bbox': {
			'x1': upscaled_cap_coords[true_obj_id]['x1'] if cap_type == 'bbox' else upscaled_ort_bbox[0],
			'y1': upscaled_cap_coords[true_obj_id]['y1'] if cap_type == 'bbox' else upscaled_ort_bbox[1],
			'x2': upscaled_cap_coords[true_obj_id]['x2'] if cap_type == 'bbox' else upscaled_ort_bbox[2],
			'y2': upscaled_cap_coords[true_obj_id]['y2'] if cap_type == 'bbox' else upscaled_ort_bbox[3]
		},
		'upscaled_ort_bbox_k': {
			'x1': upscaled_cap_coords[true_obj_id]['x1'] / w if cap_type == 'bbox' else upscaled_ort_bbox[0] / w,
			'y1': upscaled_cap_coords[true_obj_id]['y1'] / h if cap_type == 'bbox' else upscaled_ort_bbox[1] / h,
			'x2': upscaled_cap_coords[true_obj_id]['x2'] / w if cap_type == 'bbox' else upscaled_ort_bbox[2] / w,
			'y2': upscaled_cap_coords[true_obj_id]['y2'] / h if cap_type == 'bbox' else upscaled_ort_bbox[3] / h
		},
		'rect_angle': rect_angle if cap_type == 'poly' else None,
		'centroid_k': {
			'x': mask['c_x'] / w if cap_type == 'poly' else cap_centroids[true_obj_id][
				'x'],
			'y': mask['c_y'] / h if cap_type == 'poly' else cap_centroids[true_obj_id]['y']
		},
		'width_k': mask['bbox_width'] if cap_type == 'poly' else cap_params[true_obj_id]['width_k'],
		'height_k': mask['bbox_height'] if cap_type == 'poly' else cap_params[true_obj_id]['height_k'],
		'mask_points': [[p_x, p_y] for [p_x, p_y] in mask['mask_points']] if cap_type == 'poly' else None,
		'area': area if cap_type == 'poly' else (
			cap_coords[true_obj_id]['x2'] - cap_coords[true_obj_id]['x1']
		) * (cap_coords[true_obj_id]['y2'] - cap_coords[true_obj_id]['y1']),
		'side_bool': True if side_cap else False,
		'side': side_cap,
	}

	return output_caps


def dots_coords_to_list(dict_coordinates):
	"""
	Convert dict with coordinates to list

	Take dictionary with following mapping:
		{
			'top_right': {
				'x': 213,
				'y': 321
			},
			'top_left': {
				'x': 535,
				'y': 634
			},
			...
		}

	Return following list:
		[ [213, 321], [535, 634], ... ]
	"""
	return [
		[
			dict_coordinates[dot_name]['x'], dict_coordinates[dot_name]['y']
		] for dot_name in dict_coordinates.keys()
	]


def bbox_coords_to_list(dict_coordinates):
	"""
	Convert dict with bbox coordinates to list

	Take dictionary with following mapping:
		{
			'x1': 123, 'y1': 213, 'x2': 456, 'y2': 875
		}

	Return following list:
		[ [123, 213], [456, 875] ]
	"""
	return [
		[dict_coordinates['x1'], dict_coordinates['y1']],
		[dict_coordinates['x2'], dict_coordinates['y2']]
	]


def find_caps(
		mask_obj, boxes_coords, frame, obj_params, h, w,
		cap_bbox_upscale_k, draw_masks
):
	"""
	Verify ballot boxes lids to be "true".

	Return
		- frame with visualized lids
		- dict with finalized ("true") lids
	"""

	output_caps = {}
	for true_obj_id in boxes_coords.keys():

		mask = None
		if true_obj_id in mask_obj:
			mask = mask_obj[true_obj_id]

		if mask:

			# Find rotated rectangle of the lid's mask
			rect_points_np = get_rotated_bounding_rectangle(
				np.array(mask['mask_points'])
			)

			rect_points = [[int(x), int(y)] for [x, y] in rect_points_np]
			area_cap = get_area_cap(rect_points)

			# Find the closest frame side if lid is closer
			# than side_min pixels from it
			side_cap = get_side_cap(rect_points, h, w, cfg.side_min)

			# Find rotation angle of the rectangle
			rect_angle = get_rect_rotation([
				rect_points[1][0], rect_points[1][1],
				rect_points[0][0], rect_points[0][1],
				rect_points[3][0], rect_points[3][1],
				rect_points[2][0], rect_points[2][1]
			])

			# Upscale lid's rotated rectangle
			rotated_poly_upscaled_dict = upscale_poly_cap(
				rect_points,
				[
					[mask['bbox_x1'], mask['bbox_y1']],
					[mask['bbox_x2'], mask['bbox_y2']]
				]
			)

			# Upscale lid's orthogonal rectangle
			upscaled_x1, upscaled_y1, upscaled_x2, upscaled_y2 = upscale_bbox(
				mask['bbox_x1'], mask['bbox_y1'], mask['bbox_x2'], mask['bbox_y2'],
				h, w, cap_bbox_upscale_k,
				find_shifts=False
			)

			# Make output dict
			output_caps = make_output_caps_json(
				output_caps, true_obj_id, 'poly', w, h, side_cap, area_cap,
				mask=mask,
				rect_points=rect_points,
				rotated_poly_upscaled=rotated_poly_upscaled_dict,
				rect_angle=rect_angle,
				upscaled_ort_bbox=[upscaled_x1, upscaled_y1, upscaled_x2, upscaled_y2]
			)

		else:
			# If lid's mask wasn't found, use ballot box bbox for finding lid zone
			cap_coords, upscaled_cap_coords, cap_centroids, cap_params = find_caps_via_coords(
				boxes_coords, obj_params, h, w, cap_bbox_upscale_k
			)

			# Find the closest frame side if lid is closer
			# than side_min pixels from it
			side_cap = get_side_cap(cap_coords[true_obj_id], h, w, cfg.side_min)

			# Make output dict
			output_caps = make_output_caps_json(
				output_caps, true_obj_id, 'bbox', w, h, side_cap,
				cap_coords=cap_coords,
				upscaled_cap_coords=upscaled_cap_coords,
				cap_centroids=cap_centroids,
				cap_params=cap_params
			)

	if draw_masks:
		for true_obj_id, cap in output_caps.items():
			if cap['type'] == 'poly':

				rot_bbox = dots_coords_to_list(cap['rot_bbox'])
				upscaled_rot_bbox = dots_coords_to_list(cap['upscaled_rot_bbox'])

				cv2.polylines(frame, np.array([rot_bbox]), True, (21, 225, 255), 2)
				cv2.polylines(frame, np.array([upscaled_rot_bbox]), True, (0, 229, 15), 1)
				cv2.circle(
					frame,
					(int(cap['centroid_k']['x'] * w), int(cap['centroid_k']['y'] * h)),
					3, (0, 229, 15), -1
				)

			elif cap['type'] == 'bbox':

				cv2.rectangle(
					frame,
					(cap['ort_bbox']['x1'], cap['ort_bbox']['y1']),
					(cap['ort_bbox']['x2'], cap['ort_bbox']['y2']),
					(21, 225, 255), 2
				)
				cv2.rectangle(
					frame,
					(cap['ort_bbox']['x1'], cap['upscaled_ort_bbox']['y1']),
					(cap['upscaled_ort_bbox']['x2'], cap['upscaled_ort_bbox']['y2']),
					(0, 229, 15), 1
				)
				cv2.circle(
					frame,
					(int(cap['centroid_k']['x'] * w), int(cap['centroid_k']['y'] * h)),
					3, (0, 229, 15), -1
				)

	return frame, output_caps


def find_caps_via_coords(boxes_coords, obj_params, h, w, cap_bbox_upscale_k):
	"""
	Find ballot box lid zone from bbox coordinates.

	Args:
		- boxes_coords - dict with "true" ballot boxes coordinates
		- obj_params - dict with "true" ballot boxes parameters
		- h, w - height and width of resized frame
		- cap_bbox_upscale_k - upscale coefficient of lids

	Return:
		- cap_coords - dict with "true" lids coordinates
		- upscaled_cap_coords - upscaled lids coordinates
		- cap_centroids - normalized centroid coordinates
		- cap_params - ballot box lid parameters
	"""

	cap_coords = {}
	upscaled_cap_coords = {}
	cap_params = {}
	cap_centroids = {}
	for true_obj_id in boxes_coords.keys():

		width_k = obj_params[true_obj_id]['width_k']
		height_k = obj_params[true_obj_id]['height_k']
		width_ratio = width_k / height_k

		# Assume that whole ballot box bbox is a lid zone of width of a bbox > height.
		# If height > width, assume that top 40% area is a lid zone.
		if width_ratio < 1:
			cap_line_y1 = int(boxes_coords[true_obj_id][0][1] + (height_k * 0.4 * h))
		else:
			cap_line_k = width_ratio
			cap_line_y1 = int(boxes_coords[true_obj_id][0][1] * (1 + cap_line_k))
			cap_line_y1 = cap_line_y1 if cap_line_y1 < boxes_coords[true_obj_id][1][1] else \
				boxes_coords[true_obj_id][1][1]

		cap_line_y2 = cap_line_y1

		cap_x1 = boxes_coords[true_obj_id][0][0]
		cap_y1 = boxes_coords[true_obj_id][0][1]
		cap_x2 = boxes_coords[true_obj_id][1][0]
		cap_y2 = cap_line_y2

		cap_x1, cap_y1, cap_x2, cap_y2 = check_bbox_coordinates(
			cap_x1, cap_y1, cap_x2, cap_y2, h, w
		)

		cap_coords[true_obj_id] = {
			'x1': cap_x1, 'y1': cap_y1,
			'x2': cap_x2, 'y2': cap_y2
		}

		cap_c_x, cap_c_y, cap_w, cap_h = get_bbox_params(cap_x1, cap_y1, cap_x2, cap_y2)

		cap_centroids[true_obj_id] = {'x': cap_c_x / w, 'y': cap_c_y / h}
		cap_params[true_obj_id] = {'width_k': cap_w / w, 'height_k': cap_h / h}

		# Upscale lid zone by cap_bbox_upscale_k percent
		upscaled_x1, upscaled_y1, upscaled_x2, upscaled_y2 = \
			upscale_bbox(
				cap_x1, cap_y1, cap_x2, cap_y2, h, w, cap_bbox_upscale_k,
				find_shifts=False
			)

		upscaled_cap_coords[true_obj_id] = {
			'x1': upscaled_x1, 'y1': upscaled_y1,
			'x2': upscaled_x2, 'y2': upscaled_y2
		}

	return cap_coords, upscaled_cap_coords, cap_centroids, cap_params


def intersection_check(frame_rec, intersection_threshold):
	"""
	Delete highly intersected ballot boxes by other boxes
	on more than intersection_threshold percent
	"""

	if frame_rec['objects']:
		obj_num = len(frame_rec['objects'])
		obj_ids_to_delete = []

		# Fix for different key type - after json.dump all data becomes str
		id_type = None
		for obj in frame_rec['objects'].keys():
			id_type = 'int' if type(obj) is int else 'str'
			break

		# Find intersection between boxes element-wise
		for (comb_el_1, comb_el_2) in itertools.combinations(range(obj_num), 2):
			el_1 = str(comb_el_1) if id_type == 'str' else comb_el_1
			el_2 = str(comb_el_2) if id_type == 'str' else comb_el_2

			obj_1 = frame_rec['objects'][el_1]
			obj_2 = frame_rec['objects'][el_2]
			bbox_1, bbox_2 = get_bbox_coords(obj_1, obj_2)
			iou, io_bbox1, io_bbox2 = get_iou(bbox_1, bbox_2)

			max_intersection = max(io_bbox1, io_bbox2)

			if max_intersection > intersection_threshold:
				obj_id_with_max_intersection = el_1 if max_intersection == io_bbox1 else el_2
				obj_ids_to_delete.append(obj_id_with_max_intersection)

		# Delete all duplicating boxes
		obj_ids_to_delete = list(set(obj_ids_to_delete))
		for obj_id in obj_ids_to_delete:
			object_id = str(obj_id) if id_type == 'str' else obj_id
			del frame_rec['objects'][object_id]

	return frame_rec


def match_rect_with_centroid(rectangles, centroid, obj_types, obj_id):
	matched_rect = None
	for rect_coords in rectangles:

		x1, y1, x2, y2 = rect_coords[0], rect_coords[1], rect_coords[2], rect_coords[3]

		rect_c_x, rect_c_y = get_centroid(x1, y1, x2, y2)

		c_dist = get_distance_btw_dots(centroid[0], centroid[1], rect_c_x, rect_c_y)

		# obj_types[obj_id] equal to 0.0 when this ballot box
		# is invisible on the current frame
		if c_dist < 5 and obj_types[obj_id] != 0.0:
			matched_rect = (x1, y1, x2, y2)

	return matched_rect


def analyze_short_video_recs(
		recognitions, raw_masks, vid_stream_id,
		vid_frames_total, vid_path, temp_cam_dir,
		resized_width, resized_height,
		cam_quality_img_path=None,
		gpu_id=None
):
	global trackableObjects
	centroid_tracker = CentroidTracker(2000, 20)
	tracking_frame = np.zeros((resized_height, resized_width, 3), np.uint8)
	boxes_in_sight = True
	total_rec_frames = len(recognitions)

	for frame_id, frame_rec in enumerate(recognitions):

		if cfg.visualize_object_tracking:
			tracking_frame = np.zeros((resized_height, resized_width, 3), np.uint8)

		rects = []
		if frame_rec['objects']:

			# Delete highly occluded ballot boxes by other boxes
			if len(frame_rec['objects']) > 1:
				frame_rec = intersection_check(frame_rec, cfg.intersection_threshold)

			# Append ballot boxes coordinated to a list for tracking
			for obj_id in frame_rec['objects'].keys():

				obj_type = frame_rec['objects'][str(obj_id)]['class_name']
				conf = float(frame_rec['objects'][obj_id]['conf'])
				obj = frame_rec['objects'][str(obj_id)]

				bbox = get_bbox_coords(obj)
				rects.append((bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'], obj_type, conf))

		objects, obj_types = centroid_tracker.update(rects)
		objects_in_frame = objects.items()
		for (objectID, centroid) in objects_in_frame:

			to = trackableObjects.get(objectID, None)

			if to is None:
				to = TrackableObject(objectID, centroid)
			else:
				to.centroids.append(centroid)

			to.matched_in_current_frame = False

			# Match tracking boxes with boxes on the current frame
			matched_rect = match_rect_with_centroid(rects, centroid, obj_types, objectID)
			if matched_rect is not None:

				(x1, y1, x2, y2) = matched_rect

				to.visible_frames += 1
				to.visible_rate = to.visible_frames / total_rec_frames
				to.matched_in_current_frame = True
				to.coords_list.append([[x1, y1], [x2, y2]])

				_, __, obj_width, obj_height = get_bbox_params(x1, y1, x2, y2)
				area = obj_width * obj_height

				conf = obj_types[objectID][1]

				if conf > to.max_conf and to.max_conf < 0.9:
					to.max_conf = conf
					to.max_conf_frame_id = frame_id
					to.max_conf_area = area

				# If ballot box is less visible by 5%, assume that it is intersected
				if to.max_conf_area != 0:
					if abs(to.max_conf_area - area) > 0.05 * to.max_conf_area:
						to.intersected_frames += 1
						to.intersection_rate = to.intersected_frames / total_rec_frames

			# Update global dictionary
			trackableObjects[objectID] = to

			if cfg.visualize_object_tracking:
				centroid_colour = (68, 255, 87) if to.matched_in_current_frame else (20, 67, 255)
				confidence_rounded = "{:0.2f}".format(to.max_conf * 100)
				text = "# {}, {}".format(objectID, confidence_rounded)
				cv2.putText(
					tracking_frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
				)
				cv2.circle(tracking_frame, (centroid[0], centroid[1]), 4, centroid_colour, -1)

		if cfg.visualize_object_tracking:
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			cv2.imshow("Object tracking", tracking_frame)

	# Find coordinates of "true" ballot boxes
	true_objects_coords, true_objects_num, max_conf_frame_ids = \
		get_true_object_coords(vid_frames_total, tracking_frame)

	if true_objects_num == 0:
		boxes_in_sight = False
		vid_stream = cv2.VideoCapture(vid_path)
		frame_res, rec_image = get_true_frame(vid_stream, 0)
		vid_stream.release()
		return boxes_in_sight, 0, {}, {'overall_plan': {'source_frame': frame_res}}, 0.0

	else:
		# Create dictionary with ballot boxes images
		true_images = get_objects_images(
			true_objects_coords, max_conf_frame_ids,
			vid_path, recognitions, true_objects_num,
			gpu_id=gpu_id
		)

		show_true_objects = False
		if show_true_objects:
			print(true_objects_coords)
			print()
			print(true_images['overall_plan']['frame_id'])
			while True:
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				cv2.imshow('overall_plan', true_images['overall_plan']['recognized'])

			for obj_id in true_objects_coords.keys():
				print()
				print(obj_id)
				print(true_images['objects'][obj_id]['frame_id'])
				print(true_images['objects'][obj_id]['conf'])

				if cv2.waitKey(0) & 0xFF == ord('q'):
					break
				cv2.imshow('fr', true_images['objects'][obj_id]['recognized'])

		# Classify ballot boxes type based on lids
		object_types, mask = recognize_objects_types(
			true_images,
			out_path=temp_cam_dir,
			gpu_id=gpu_id
		)

		# Compute camera "quality" parameter
		frame, cam_quality, obj_params = analyze_cam_quality(
			true_images, true_objects_coords, object_types,
			resized_height, resized_width,
			draw_boxes=cfg.save_bboxes_quality_image
		)

		# Find "true" lids and visualize them
		frame, output_caps = find_caps(
			mask, true_objects_coords, frame, obj_params,
			resized_height, resized_width,
			cfg.cap_bbox_upscale_k,
			cfg.save_bboxes_quality_image
		)

		# Take all gathered docs to output_true_objects and output_true_images dicts
		output_true_objects = {}
		output_true_images = {'objects': {}}
		for true_obj_id in true_objects_coords.keys():
			output_true_objects[true_obj_id] = {
				'bbox': {
					'x1': true_objects_coords[true_obj_id][0][0],
					'y1': true_objects_coords[true_obj_id][0][1],
					'x2': true_objects_coords[true_obj_id][1][0],
					'y2': true_objects_coords[true_obj_id][1][1]
				},
				'bbox_k': {
					'x1': true_objects_coords[true_obj_id][0][0] / resized_width,
					'y1': true_objects_coords[true_obj_id][0][1] / resized_height,
					'x2': true_objects_coords[true_obj_id][1][0] / resized_width,
					'y2': true_objects_coords[true_obj_id][1][1] / resized_height
				},
				'caps': output_caps[true_obj_id],
				'centroid_k': {
					'x': obj_params[true_obj_id]['centroid_k'][0],
					'y': obj_params[true_obj_id]['centroid_k'][1]
				},
				'obj_quality': obj_params[true_obj_id]['object_quality'],
				'conf': obj_params[true_obj_id]['conf'],
				'type': object_types[true_obj_id]['type'],
				'type_conf': object_types[true_obj_id]['type_conf'],
				'obj_params': {
					'distance_from_center_k': obj_params[true_obj_id]['distance_from_center'],
					'width_k': obj_params[true_obj_id]['width_k'],
					'height_k': obj_params[true_obj_id]['height_k'],
					'area_k': obj_params[true_obj_id]['area_k'],
					'visible_rate': obj_params[true_obj_id]['visible_rate'],
					'intersection_rate': obj_params[true_obj_id]['intersection_rate'],
					'normalized_dist_k': obj_params[true_obj_id]['normalized_dist_k'],
					'normalized_width_k': obj_params[true_obj_id]['normalized_width_k']
				},
			}

			output_true_images['objects'][true_obj_id] = {
				'source_frame': true_images['objects'][true_obj_id]['source'],
				'recognized_frame': true_images['objects'][true_obj_id]['recognized']
			}

		output_true_images['overall_plan'] = {
			'source_frame': true_images['overall_plan']['source'],
			'recognized_frame': true_images['overall_plan']['recognized'],
			'processed_frame': frame
		}

		if cfg.save_bboxes_quality_image:
			cv2.imwrite(
				'{}/{}_{}.png'.format(
					temp_cam_dir, vid_stream_id, int(cam_quality * 100)
				),
				frame
			)
		if cam_quality_img_path:
			cv2.imwrite(cam_quality_img_path, frame)

		if cfg.visualize_bboxes_quality:
			while True:
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				cv2.imshow('window', frame)

			cv2.destroyAllWindows()

		return boxes_in_sight, true_objects_num, output_true_objects, \
			output_true_images, cam_quality


def main():

	camera_num = 1
	reg_uik = '1_12345'
	video_path = 'path_to_video'

	temp_dir = f'data/temp/{reg_uik}'

	# cam_quality_image_path = '123.jpg'
	cam_quality_image_path = None
	boxes, objects_num, objects, objects_images, camera_quality = find_cam_quality(
		camera_num=camera_num,
		vid=video_path,
		temp_camera_dir=temp_dir,
		start_second=0,
		verbose=True,
		camera_quality_img_path=cam_quality_image_path
	)
	print(boxes)
	print(objects_num)
	print(objects)
	print(camera_quality)


if __name__ == '__main__':
	main()




