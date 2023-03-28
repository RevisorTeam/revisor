import itertools

from cfg import cfg
import json
import csv
import cv2
import os
from math import hypot
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, LineString, LinearRing
import multiprocessing
from shapely import affinity
import math
from math import pi
from itertools import groupby
from operator import itemgetter


#########################################
# Geometry utils
#########################################
def voter_stopped_near_box(
		ballot_boxes, cap_centroids_coords,
		accepted_box_distances, centroids, cur_centroid,
		voter_width, image
):
	"""
	Detect person stopping near a ballot box.

	Return:
		- stopped_near_box:
			* True - if a person has stopped near a ballot box
			* False - if not
	"""

	stopped_near_box, near_box = False, False

	# If a tracker has less than min_frames_to_stop processed frames,
	# skip it
	if len(centroids) < cfg.min_frames_to_stop:
		return stopped_near_box, near_box

	for ballot_box_id, ballot_box_dict in ballot_boxes.items():

		# Find distance between person centroid
		# and ballot box centroid
		dist_to_box = get_distance_btw_dots(
			cap_centroids_coords[ballot_box_id]['x'],
			cap_centroids_coords[ballot_box_id]['y'],
			cur_centroid['x'], cur_centroid['y']
		)

		# Determine whether person centroid is inside stopping zone.
		# Stopping zone is a circle with radius = ballot_box_width * 120%
		accepted_dist = accepted_box_distances[ballot_box_id]

		if dist_to_box < accepted_dist:
			near_box = True
			break

	# If a person was near a ballot box, determine
	# whether person stopped near it
	if near_box:

		# Find average centroid of last min_frames_to_stop centroid coordinates
		last_centroids = np.array(
			[[c['x'], c['y']] for c in centroids[-cfg.min_frames_to_stop:]]
		)
		averaged_centroid = np.average(last_centroids, axis=0).tolist()

		if cfg.show_recognitions:
			cv2.circle(
				image,
				(int(averaged_centroid[0]), int(averaged_centroid[1])),
				5, (0, 0, 255), 1
			)

		# Find distance between average centroid of last N coordinates
		# and current centroid
		avg_to_cur_dist = get_distance_btw_dots(
			averaged_centroid[0], averaged_centroid[1],
			cur_centroid['x'], cur_centroid['y']
		)

		# Detect stopping near ballot box
		if avg_to_cur_dist < int(voter_width * cfg.avg_voter_centroid_k):
			# Last N averaged centroid coordinates
			# will not change when a person will stand still
			stopped_near_box = True

	return stopped_near_box, near_box


def check_bbox_coordinates(x1, y1, x2, y2, h, w):
	"""
	Make sure that input points (x1, y1) and (x2, y2)
	don't go out of the frame
	"""
	new_x1 = 0 if x1 < 0 else x1
	new_y1 = 0 if y1 < 0 else y1
	new_x2 = w if x2 > w else x2
	new_y2 = h if y2 > h else y2
	return new_x1, new_y1, new_x2, new_y2


def coefficient_to_coordinate(coefficient: float, side: int):
	return int(coefficient * side)


def coefficients_to_coordinates(coefficients: dict, width: int, height: int):
	"""
	Convert normalized point coordinates to source coordinates.

	Normalized coordinates had been computed as a ratio
	of X or Y value on frame's Width or Height respectively.
	"""
	coordinates = {}
	for coef_name, coef_value in coefficients.items():
		side = width if 'x' in coef_name else height
		coordinates[coef_name] = coefficient_to_coordinate(coef_value, side)
	return coordinates


def get_side_cap(point, h, w, dist):
	"""
	Find the nearest frame side names.

	:param point: inputs points of a ballot box lid
	:param h: frame height
	:param w: frame width
	:param dist: distance to a frame side

	:return: List of the closest frame sides.
	"""

	result = set()
	if isinstance(point, dict):
		if point['x1']-dist < 0:
			result.add('left')
		if point['x2']+dist > w:
			result.add('right')
		if point['y1']-dist < 0:
			result.add('top')
		if point['y2']+dist > h:
			result.add('down')
	else:
		for pt in point:
			if pt[0]-dist < 0:
				result.add('left')
			if pt[0]+dist > w:
				result.add('right')
			if pt[1]-dist < 0:
				result.add('top')
			if pt[1]+dist > h:
				result.add('down')

	return list(result) if len(result) else None


def upscale_poly_cap(rect_points: list, points_bbox: list):
	"""
	Upscale input rotated rectangle
	by cfg.cap_rotated_bbox_upscale_k percent.

	Args:
		rect_points: List of a rotated rectangle coordinates:
			[ [x1, y1], ..., [x4, y4] ]
		points_bbox: List of an orthogonal rectangle coordinates:
			[ [x1, y1], [x2, y2] ]

	Return:
		Dictionary with upscaled rotated rect coordinates
		with a following mapping:
			{ 'top_right': { 'x': ..., 'y': ... }, ... }
	"""

	# Find approximate width and height of a lid
	_, __, bbox_width, bbox_height = \
		get_bbox_params(
			points_bbox[0][0], points_bbox[0][1],
			points_bbox[1][0], points_bbox[1][1]
		)

	rotated_poly = Polygon(rect_points)

	# Upscale rect.
	# Upscaling size of a max side equals to an upscaling size of a min side
	max_side = max(bbox_width, bbox_height)
	min_side = min(bbox_width, bbox_height)
	ratio_k = float(max_side / min_side)

	max_side_k = 1 + cfg.cap_rotated_bbox_upscale_k
	min_side_k = 1 + cfg.cap_rotated_bbox_upscale_k * ratio_k

	if bbox_width >= bbox_height:
		rotated_poly_upscaled_np = affinity.scale(
			rotated_poly, xfact=max_side_k, yfact=min_side_k
		)
	else:
		rotated_poly_upscaled_np = affinity.scale(
			rotated_poly, xfact=min_side_k, yfact=max_side_k
		)

	# [:4] is needed to remove last (fifth) coordinate.
	# By default, 5 coordinates are returned. Fifth coords equals to a first one
	rotated_poly_upscaled = [
		[int(x), int(y)] for (x, y) in list(rotated_poly_upscaled_np.exterior.coords)[:4]
	]

	return {
		"top_right": {"x": rotated_poly_upscaled[0][0], "y": rotated_poly_upscaled[0][1]},
		"top_left": {"x": rotated_poly_upscaled[1][0], "y": rotated_poly_upscaled[1][1]},
		"bottom_left": {"x": rotated_poly_upscaled[2][0], "y": rotated_poly_upscaled[2][1]},
		"bottom_right": {"x": rotated_poly_upscaled[3][0], "y": rotated_poly_upscaled[3][1]}
	}


def upscale_bbox_cap(points_bbox, resized_width, resized_height):
	"""
	Upscale input bounding box by cfg.cap_bbox_upscale_k percent
	"""
	upscaled_x1, upscaled_y1, upscaled_x2, upscaled_y2 = \
		upscale_bbox(
			points_bbox['x1'], points_bbox['y1'], points_bbox['x2'], points_bbox['y2'],
			resized_height, resized_width, cfg.cap_bbox_upscale_k, find_shifts=False
		)
	return {"x1": upscaled_x1, "y1": upscaled_y1, "x2": upscaled_x2, "y2": upscaled_y2}


def upscale_bbox(x1, y1, x2, y2, height, width, bbox_upscale_k, find_shifts=False):
	"""
	Upscale bbox coordinates (not the coefficients) by bbox_upscale_k percent.

	If "find_shifts" set to true, upscaling values (in pixels) will be found for each
	of the directions.
	That is needed for finding source (input) coordinates.

	Args:
		x1, y1,	x2, y2: bbox coordinates (int)
		height, width: height and width of the frame
		bbox_upscale_k: upscaling percent og bbox (from 0.0 to 1.0) (float)
		find_shifts: whether to return shifts for each upscaling side
	"""

	_, __, obj_width, obj_height = get_bbox_params(x1, y1, x2, y2)
	width_ratio_k = obj_height / obj_width

	# Find width and height shifts.
	# Width shift equals to height shift (if it's enough room for that)
	width_shift = int(obj_width * bbox_upscale_k * width_ratio_k)
	height_shift = int(obj_height * bbox_upscale_k)

	# Find upscaled coordinates (some coordinates may lay outside the frame)
	new_x1 = x1 - width_shift
	new_y1 = y1 - height_shift
	new_x2 = x2 + width_shift
	new_y2 = y2 + height_shift

	# Make sure all coordinates lay inside the frame
	upscaled_x1, upscaled_y1, upscaled_x2, upscaled_y2 = \
		check_bbox_coordinates(
			new_x1, new_y1, new_x2, new_y2,
			h=height, w=width
		)

	if find_shifts:

		x_left_shift = width_shift
		x_right_shift = width_shift
		y_bottom_shift = height_shift
		y_top_shift = height_shift

		if new_x1 != upscaled_x1:
			x_left_shift = width_shift - abs(new_x1)
		if new_x2 != upscaled_x2:
			x_right_shift = width_shift - abs(width - new_x2)
		if new_y1 != upscaled_y1:
			y_bottom_shift = height_shift - abs(new_y1)
		if new_y2 != upscaled_y2:
			y_top_shift = height_shift - abs(height - new_y2)

		return upscaled_x1, upscaled_y1, upscaled_x2, upscaled_y2, {
			'x_left_shift': x_left_shift, 'x_right_shift': x_right_shift,
			'y_bottom_shift': y_bottom_shift, 'y_top_shift': y_top_shift
		}
	else:
		return upscaled_x1, upscaled_y1, upscaled_x2, upscaled_y2


def get_bbox_params(x1, y1, x2, y2, out_type='float'):
	"""
	Compute basic bbox parameters:
		* width
		* height
		* centroid coordinates
	"""

	x1_bbox = x1
	y1_bbox = y1
	x2_bbox = x2
	y2_bbox = y2

	width_bbox = x2_bbox - x1_bbox
	height_bbox = y2_bbox - y1_bbox

	# Bounding box centroid
	c_x_bbox = x1_bbox + width_bbox / 2
	c_y_bbox = y1_bbox + height_bbox / 2

	if out_type == 'float':
		return c_x_bbox, c_y_bbox, width_bbox, height_bbox
	elif out_type == 'int':
		return {'x': int(c_x_bbox), 'y': int(c_y_bbox)}, int(width_bbox), int(height_bbox)


def get_bbox_coords(object_1, object_2=None):
	bbox_1 = {
		'x1': int(object_1['coords'][0][0]), 'x2': int(object_1['coords'][1][0]),
		'y1': int(object_1['coords'][0][1]), 'y2': int(object_1['coords'][1][1]),
	}
	if object_2:
		bbox_2 = {
			'x1': int(object_2['coords'][0][0]), 'x2': int(object_2['coords'][1][0]),
			'y1': int(object_2['coords'][0][1]), 'y2': int(object_2['coords'][1][1]),
		}
		return bbox_1, bbox_2
	else:
		return bbox_1


def get_centroid(x1, y1, x2, y2):
	"""
	Compute bbox centroid
	"""

	x1_bbox = int(x1)
	y1_bbox = int(y1)
	x2_bbox = int(x2)
	y2_bbox = int(y2)

	width_bbox = x2_bbox - x1_bbox
	height_bbox = y2_bbox - y1_bbox

	c_x_bbox = x1_bbox + width_bbox / 2
	c_y_bbox = y1_bbox + height_bbox / 2

	return int(c_x_bbox), int(c_y_bbox)


def get_rotated_bounding_rectangle(contours):
	"""
	Find bounding rectangle of an input mask.

	:param contours: Mask coordinates

	:return: Coordinates of rotated bounding rectangle
	"""
	max_dist = {'p1': 0, 'p2': 0, 'dist': 0}
	for (p1, p2) in itertools.combinations(range(len(contours)), 2):
		dist = math.hypot(
			contours[p2][0] - contours[p1][0],
			contours[p2][1] - contours[p1][1]
		)
		if dist > max_dist['dist']:
			max_dist = {'p1': contours[p1], 'p2': contours[p2], 'dist': dist}

	p1 = max_dist['p1']
	p2 = max_dist['p2']

	k = (p1[1] - p2[1]) / (p1[0] - p2[0])
	b = p2[1] - k * p2[0]
	dist = lambda x, y: (k * x + b - y) / (math.sqrt(k ** 2 + 1))
	max_dist_p1 = {'p': 0, 'dist': 0}
	min_dist_p1 = {'p': 0, 'dist': 0}

	for point in contours:
		dst = dist(point[0], point[1])
		if dst > max_dist_p1['dist']:
			max_dist_p1 = {'p': point, 'dist': dst}

		if dst < min_dist_p1['dist']:
			min_dist_p1 = {'p': point, 'dist': dst}

	return [max_dist['p1'], max_dist_p1['p'], max_dist['p2'], min_dist_p1['p']]


def get_area_cap(coords):
	"""
	Compute area of an input mask.

	:param coords: mask coordinates (polygon)

	:return: mask area in pixels
	"""
	return Polygon([[p[0], p[1]] for p in coords]).area


def get_bbox_from_coords(points):
	"""
	Find bounding box from mask coordinates
	"""
	x_coordinates, y_coordinates = zip(*points)
	return [
		int(min(x_coordinates)), int(min(y_coordinates)),
		int(max(x_coordinates)), int(max(y_coordinates))
	]


def get_rect_rotation(rect_coords):
	"""
	Compute rotation angle of an rectangle from its coordinates:
	[x1, y1, x2, y2, x3, y3, x4, y4], where:
		top left    : x1, y1
		top right   : x2, y2
		bottom right: x3, y3
		bottom left : x4, y4
	"""
	diffs = [rect_coords[0] - rect_coords[6], rect_coords[1] - rect_coords[7]]
	diffs[1] = diffs[1] if diffs[1] != 0 else 0.00001
	rotation = math.atan(diffs[0] / diffs[1]) * 180 / pi

	return rotation


def get_iou(bb1, bb2):
	"""
	Compute following:
		* intersection over union of two input bboxes
		* the ratio of the intersection of two bboxes
			to the area of the first bbox
		* the ratio of the intersection of two bboxes
			to the area of the second bbox
	"""
	assert bb1['x1'] < bb1['x2']
	assert bb1['y1'] < bb1['y2']
	assert bb2['x1'] < bb2['x2']
	assert bb2['y1'] < bb2['y2']

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1['x1'], bb2['x1'])
	y_top = max(bb1['y1'], bb2['y1'])
	x_right = min(bb1['x2'], bb2['x2'])
	y_bottom = min(bb1['y2'], bb2['y2'])

	if x_right < x_left or y_bottom < y_top:
		return 0.0, 0.0, 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
	bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	io_bbox1 = intersection_area / float(bb1_area)
	io_bbox2 = intersection_area / float(bb2_area)
	assert iou >= 0.0
	assert iou <= 1.0
	assert io_bbox1 >= 0.0
	assert io_bbox1 <= 1.0
	assert io_bbox2 >= 0.0
	assert io_bbox2 <= 1.0
	return iou, io_bbox1, io_bbox2


def get_distance_btw_dots(x1, y1, x2, y2):
	"""
	Compute distance between two input points
	"""
	return hypot(x2-x1, y2-y1)


def parabolic_func(x):
	return x**2


#########################################
# Directories and files utils
#########################################
def read_json(json_path):
	json_file = {}
	if os.path.isfile(json_path):
		with open(json_path) as f:
			json_file = json.load(f)
	return json_file


def read_csv(csv_path):
	out_csv = []
	if os.path.isfile(csv_path):
		with open(csv_path, newline='') as f:
			reader = csv.reader(f)
			for row in reader:
				out_csv.extend(row)
	return out_csv


def append_row_to_csv(csv_path, row_list):
	with open(csv_path, 'a') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerows([row_list])


def save_output_json(path, output_dict, save_json=True):
	if save_json:
		with open(path, "w") as j:
			json.dump(output_dict, j, indent=2)


def create_folder(path):
	"""
	Recursively create target folder
	"""
	if not os.path.exists(path):
		os.makedirs(path)


def get_camera_info(directory):
	"""
	Parse from directory name:
		* region number (region_num)
		* station number (uik_num)
		* camera number (camera_num)
		* camera identificator
	"""
	split_dir = directory.split('_')
	tmp_info, camera_id = split_dir[:2]
	tmp_info, camera_num = tmp_info.split('c')
	tmp_info, uik_num = tmp_info.split('u')
	tmp_info, region_num = tmp_info.split('r')
	return int(region_num), int(uik_num), int(camera_num), camera_id


#########################################
# Other
#########################################
def get_yolo_trt_bboxes(
		pred_bboxes: np.array,
		pred_confidences: np.array,
		pred_classes_ids: np.array,
		yolo_classes: dict
) -> np.array:
	"""
	Transforms TensorRT output data to the imput data format of SORT tracker

	Args:
		pred_bboxes: Bbox coordinates nparray. Each element is array of:
			[x1, y1, x2, y2, cls_id, prob]
		pred_confidences: nparray (prob) - np array with confidences of predicted bboxes
		pred_classes_ids: nparray (prob) - np array with predicted classes ids
		yolo_classes: Dictionary with mapping:
			(id_int: class_name_str)

	Returns:
		nparray (x1, y1, x2, y2, score)
	"""
	bboxes = []
	for object_id, bbox in enumerate(pred_bboxes):
		class_name = yolo_classes[int(pred_classes_ids[object_id])]
		if class_name != 'person':
			continue
		bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], pred_confidences[object_id]])
	return np.array(bboxes) if bboxes else np.empty((0, 5))


def detect_hardware():
	import tensorflow as tf
	cpu_count = multiprocessing.cpu_count()
	try:
		tf_gpus = tf.config.experimental.list_physical_devices('GPU')
	except:
		tf_gpus = []

	gpus = [gpu_id for gpu_id, gpu in enumerate(tf_gpus)]
	return list(gpus), tf_gpus, cpu_count


def get_filled_sequences(ids: list):
	"""
	Find sets of consecutive numbers within a given list of numbers.

	For example:
		ids = [1, 4, 5, 6, 10, 15, 16, 17, 18, 22, 25, 26, 27, 28]
		filled_sequences = [[1], [4, 5, 6], [10], [15, 16, 17, 18], [22], [25, 26, 27, 28]]

	Params: ids - list of ordered (asc) int numbers

	Return: filled_sequences - sets of consecutive numbers
	"""
	filled_sequences = []
	for k, g in groupby(enumerate(ids), lambda ix: ix[0] - ix[1]):
		filled_sequences.append(list(map(itemgetter(1), g)))
	return filled_sequences


def average(lst):
	"""
	Compute average of a list
	"""
	return sum(lst) / len(lst)


def chunk_list(seq, num):
	avg = len(seq) / float(num)
	out = []
	last = 0.0
	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg
	return out


def del_ballot_boxes(ballot_boxes):
	"""
	Delete ballot boxes from recognized dictionary.
	In case when there are 'koib' ballot box type.
	"""
	are_koibs = False
	boxes_ids_to_pop = []
	for ballot_box_id, ballot_box_dict in ballot_boxes.items():

		if ballot_box_dict['type'] == 'koib':
			are_koibs = True

		if ballot_box_dict['type'] == 'ballot_box':
			boxes_ids_to_pop.append(ballot_box_id)

	if are_koibs:
		for ballot_box_id in boxes_ids_to_pop:
			ballot_boxes.pop(ballot_box_id)

	return ballot_boxes


def get_avg_box_coefs(box_dict):
	dist_values, width_values, = [], []
	for box_id, box in box_dict.items():
		dist_values.append(box['obj_params']['normalized_dist_k'])
		width_values.append(box['obj_params']['normalized_width_k'])

	return average(dist_values), average(width_values)


def get_voting_day_times(video_info):

	start_hour, start_min, start_sec, end_hour, end_min, end_sec = \
		video_info['start_time'].hour, video_info['start_time'].minute, video_info['start_time'].second, \
		video_info['end_time'].hour, video_info['end_time'].minute, video_info['end_time'].second
	official_start_hour, official_end_hour = cfg.approved_hours['start'], cfg.approved_hours['end']

	# Find number of seconds to skip if video starts
	# earlier than polling station opening hour
	seconds_to_skip = 0
	if start_hour < official_start_hour:
		start_hours_diff = official_start_hour - start_hour - 1
		start_minutes_diff = 60 - start_min
		start_seconds_diff = 60 - start_sec
		seconds_to_skip = \
			start_hours_diff * 3600 + start_minutes_diff * 60 + start_seconds_diff
		print('Skipping time: {}:{}:{}, seconds to skip: {}'.format(
			start_hours_diff, start_minutes_diff,
			start_seconds_diff, seconds_to_skip
		))

	# Find earlier stopping time if video ends after closing hour.
	# 	* If video ends the next day, add 24 hours - video end time
	# 	must be later than the start time
	end_hour = 24 + end_hour if start_hour > end_hour else end_hour
	stop_time = 0
	if end_hour > official_end_hour:
		# Don't subtract 1 from end_hours_diff,
		# as cfg.approved_hours['end'] set to 19 (not to 20)
		end_hours_diff = official_end_hour - start_hour
		end_minutes_diff = 60 - start_min
		end_seconds_diff = 60 - start_sec
		stop_time = end_hours_diff * 3600 + end_minutes_diff * 60 + end_seconds_diff
		print(
			'Video will be early stopped! '
			'Time till end: {}:{}:{}, stop time seconds: {}'.format(
				end_hours_diff, end_minutes_diff, end_seconds_diff, stop_time
			)
		)

	return seconds_to_skip, stop_time


def get_voted_ballot_box(
		joints_dict, cap_centroids_coords, voting_start, voting_end, box_intersections
):
	"""
	Find the closest ballot box during voting action.
	Hands coordinates and lid zone are used for that purpose
	"""
	avg_distances = {}
	for ballot_box_id, cap_coords in cap_centroids_coords.items():

		total_distance = 0.0
		frames_with_joints = joints_dict.keys()
		for frame_id in range(voting_start, voting_end + 1):
			if frame_id in frames_with_joints:
				if joints_dict[frame_id]:
					total_distance += get_distance_btw_dots(
						cap_coords['x'], cap_coords['y'],
						joints_dict[frame_id]['left_wrist'][0],
						joints_dict[frame_id]['right_wrist'][1]
					)

		# Find average distance from person hands to each ballot box
		avg_distances[ballot_box_id] = total_distance / box_intersections[ballot_box_id] \
			if ballot_box_id in box_intersections.keys() else float('Inf')

	# Get ballot box id with the smallest average distance
	voted_ballot_box_id = min(avg_distances, key=avg_distances.get)

	return voted_ballot_box_id


def replace_dict_frame_ids(
		input_dict, min_frame_id, max_frame_id, voting_start, voting_end
):
	"""
	Make output sample dictionary of an input dict and
	update frame_ids from source to output ones.

	For example:
	Inputs:
		input_dict.keys() = [349, 350, 351, ... 490]
		min_frame_id = 360
		max_frame_id = 450
		voting_start = 380
		voting_end = 430

	Outputs:
		new_frames.keys() = [0, 1, 2, ... 90]
		new_voting_start = 20
		new_voting_end = 70

	Args:
		input_dict: dict with bboxes / poses / confs / orientations data of a person.
			Dictionary has source frame ids in keys
		min_frame_id: starting source frame id of an output video
		max_frame_id: ending source frame id of an output video
		voting_start: source frame id of starting voting action
		voting_end: source frame id of ending voting action

	Return:
		new_frames: input dict with updated frame ids and target frames values
		new_voting_start: updated frame_id of voting action starting
		new_voting_end: updated frame_id of voting action ending
	"""

	saving_frame_ids = list(range(min_frame_id, max_frame_id + 1))
	old_frames_list = list(input_dict.keys())

	new_frames = {}
	new_voting_start = 0
	new_voting_end = 0
	for new_frame_id, old_frame_id in enumerate(saving_frame_ids):

		new_frames[str(new_frame_id)] = input_dict[old_frame_id] \
			if old_frame_id in old_frames_list else {}

		if old_frame_id == voting_start:
			new_voting_start = new_frame_id
		if old_frame_id == voting_end:
			new_voting_end = new_frame_id

	return new_frames, new_voting_start, new_voting_end


def find_station_by_cam_id(cam_id, stations):
	"""
	Find polling station record by camera id
	"""
	for station in stations:
		if cam_id in station['camera_id']:
			return station
	return None
