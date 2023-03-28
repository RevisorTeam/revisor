from shapely.geometry import LineString
import math
from math import pi
from cfg import cfg
from utils import get_filled_sequences


abs_names_converter = {
	'back_side': {
		cfg.abs_side_names[0]: cfg.direction_names[6],
		cfg.abs_side_names[1]: cfg.direction_names[7],
		cfg.abs_side_names[2]: cfg.direction_names[0],
		cfg.abs_side_names[3]: cfg.direction_names[1],
		cfg.abs_side_names[4]: cfg.direction_names[2]
	},
	'front_side': {
		cfg.abs_side_names[0]: cfg.direction_names[6],
		cfg.abs_side_names[1]: cfg.direction_names[5],
		cfg.abs_side_names[2]: cfg.direction_names[4],
		cfg.abs_side_names[3]: cfg.direction_names[3],
		cfg.abs_side_names[4]: cfg.direction_names[2]
	},
}


def get_person_orientation(joints, width, height):
	"""
	Get relative to the camera person orientation.

	Args:
		joints: dictionary with 17 coco joints.
			Dict value is a list [x, y]
		width: frame width.
		height: frame height.

	Return:

		1) Orientation zone names:

		|   back_left   |   back   |   back_right   |
		|   left        |          |   right        |
		|   front_left  |   front  |   front_right  |

		back - person directed with his / her back to the camera
		front - person directed towards the camera

		2) orientation ange (varies from 0 to 360 degree, int):

		Angle between oX axis and person direction line

				| 90
				|
		0		|
		__________________
		360		|		180
				|
			270	|

		3) coordinates of the perpendicular line drawn from
			the middle of the shoulders
	"""

	shoulder_line = LineString([
		(joints['left_shoulder'][0], joints['left_shoulder'][1]),
		(joints['right_shoulder'][0], joints['right_shoulder'][1])
	])

	# Find orthogonal line to the shoulders line
	right_parallel_line = shoulder_line.parallel_offset(shoulder_line.length, 'right')

	# Get the direction line
	shoulder_line_centroid = shoulder_line.centroid
	parallel_line_centroid = right_parallel_line.centroid
	orthogonal_line = LineString([
		(shoulder_line_centroid.x, shoulder_line_centroid.y),
		(parallel_line_centroid.x, parallel_line_centroid.y)
	])

	# Find the angle between:
	# - perpendicular to the shoulder line (outgoing from the center of the shoulder line)
	# - a line parallel to the X axis and passing through the center of the shoulder line
	# For cases where the Y-coordinates of the shoulders are equal, assign the ratio of the line
	# shoulders to frame width (for escaping division by zero)
	orth_line_coords = list(orthogonal_line.coords)
	x_axis_line_coords = [(0, orth_line_coords[0][1]), (width, orth_line_coords[0][1])]

	m1 = (orth_line_coords[1][1] - orth_line_coords[0][1]) / (orth_line_coords[1][0] - orth_line_coords[0][0]) \
		if orth_line_coords[1][0] - orth_line_coords[0][0] != 0.0 else width
	m2 = (x_axis_line_coords[1][1] - x_axis_line_coords[0][1]) / (x_axis_line_coords[1][0] - x_axis_line_coords[0][0]) \
		if x_axis_line_coords[1][0] - x_axis_line_coords[0][0] != 0.0 else width

	angle_rad = abs(math.atan(m1) - math.atan(m2))
	angle_deg = angle_rad * 180 / pi

	# Find person orientation name and angle
	orientation, angle_360 = get_orientation_direction(orth_line_coords, angle_deg)

	direction_line = {

		# Middle of the shoulders point
		'shoulders_mid_x1': orth_line_coords[0][0],
		'shoulders_mid_y1': orth_line_coords[0][1],

		# Point of the perpendicular to the shoulders line
		'dir_x2': orth_line_coords[1][0],
		'dir_y2': orth_line_coords[1][1],
	}

	return {
		'orientation': orientation,
		'angle': angle_360,
		'direction_line': direction_line
	}


def get_abs_horizontal_direction(
		center_point_x,
		direction_point_x,
		x_axis_angle,
		half_angle_step,
		abs_side_names
):
	"""
	Find horizontal direction of a person.

	half_angle_step = 22.5 degree

	x_axis_angle - angle between person direction line and oX axis (0 ... 90)
		- If a direction line directed to the left, angle decreases to the left side of oX
		- If a direction line directed to the right, angle decreases to the right side of oX

				90  |  90 degree
					|
		0			|	  	   0 degree
		________________________

	Positive oY axis consists of 5 parts:
		- left - from 0 to 22.5 degree
		- top_left - from 22.5 to 67.5 degree
		- top - from 67.5 to 90 (for the left side), from 90 to 67.5 (for the right side)
		- top_right - from 67.5 to 22.5
		- right - from 22.5 to 0
	"""
	abs_horizontal_direction = ''
	if direction_point_x <= center_point_x:
		horizontal_side = 'left'
		if 0.0 <= x_axis_angle <= half_angle_step:
			abs_horizontal_direction = abs_side_names[0]

		elif half_angle_step < x_axis_angle <= 3 * half_angle_step:
			abs_horizontal_direction = abs_side_names[1]

		elif 3 * half_angle_step < x_axis_angle <= 90.0:
			abs_horizontal_direction = abs_side_names[2]

	else:
		horizontal_side = 'right'
		if 90.0 >= x_axis_angle > 3 * half_angle_step:
			abs_horizontal_direction = abs_side_names[2]

		elif 3 * half_angle_step >= x_axis_angle > half_angle_step:
			abs_horizontal_direction = abs_side_names[3]

		elif half_angle_step >= x_axis_angle >= 0.0:
			abs_horizontal_direction = abs_side_names[4]

	return abs_horizontal_direction, horizontal_side


def get_orientation_direction(direction_line, x_axis_angle):
	"""
	Get orientation name (1 of 8 possible names) and orientation angle (0 ... 360)
	based on direction line coordinates.
	"""

	center_point_x, center_point_y, direction_point_x, direction_point_y = \
		direction_line[0][0], direction_line[0][1], \
		direction_line[1][0], direction_line[1][1]

	# Direction names consists of 8 parts. Each part is a 45 degree zone
	angle_step = float(360 / len(cfg.direction_names))
	half_angle_step = angle_step / 2  		# 22.5 degrees

	# Find horizontal person direction
	abs_horizontal_direction, horizontal_side = get_abs_horizontal_direction(
		center_point_x, direction_point_x, x_axis_angle,
		half_angle_step, cfg.abs_side_names
	)

	# Find vertical person direction and quadrant (to find angle later)
	if direction_point_y < center_point_y:
		quarter = 1 if horizontal_side == 'left' else 2
		orientation = abs_names_converter['back_side'][abs_horizontal_direction]
	else:
		quarter = 4 if horizontal_side == 'left' else 3
		orientation = abs_names_converter['front_side'][abs_horizontal_direction]

	# Compute person orientation angle (0 ... 360)
	angle_360 = 0
	if quarter == 1:
		angle_360 = x_axis_angle
	elif quarter == 2:
		angle_360 = 90.0 + (90.0 - x_axis_angle)
	elif quarter == 3:
		angle_360 = 180.0 + x_axis_angle
	elif quarter == 4:
		angle_360 = 360.0 - x_axis_angle

	return orientation, angle_360


def get_voting_orientation(orientation_dict, voting_start_frame_id, voting_end_frame_id):
	"""
	Find most common person orientation name during the voting action
	"""
	orientations_names = {}

	for frame_id, orientation in orientation_dict.items():
		if orientation and voting_start_frame_id < int(frame_id) < voting_end_frame_id:

			if orientation['orientation'] not in orientations_names.keys():
				orientations_names[orientation['orientation']] = 1
			else:
				orientations_names[orientation['orientation']] += 1

	most_common_orientation = max(
		orientations_names,
		key=orientations_names.get
	) if orientations_names else ''

	return {
		'voting_orientation': most_common_orientation
	}


def interpolate_bboxes(bboxes):
	"""
	Fill missing bbox coordinates utilizing linear interpolation:
		1. For input person bboxes build two lines:
			- line with upper left coordinates of the first and last non-empty frames
			- line with the lower right coordinates of the first and last non-empty frames
		2. Divide each line into X parts (depending on the number of dropped frames)
		3. Find missing values of the coordinates of each part of the line
	"""

	# Find frame_ids of missing bboxes
	existing_bbox_frame_ids = [frame_id for frame_id, bbox in bboxes.items() if bbox]
	filled_sequences = get_filled_sequences(existing_bbox_frame_ids)
	seq_total = len(filled_sequences)

	ids_to_interpolate = [
		[max(seq), min(filled_sequences[seq_id+1])] for seq_id, seq in enumerate(
			filled_sequences) if seq_id != (seq_total-1)
	]

	# Iterate over sequences with empty frame_ids:
	# 	* first_id - first non-empty frame in the sequence
	# 	* last_id - last non-empty frame in the sequence
	# all frames in between are empty
	for [first_id, last_id] in ids_to_interpolate:

		first_bbox = bboxes.get(first_id, {})
		last_bbox = bboxes.get(last_id, {})

		# Get sequence's frame_ids to interpolate (empty ones)
		interpolating_ids = list(range(first_id + 1, last_id))

		interpolation_step = 1 / (len(interpolating_ids) + 1)

		# Define interpolating lines:
		# 	* top_left_line is an averaging track of top left bbox point
		# 	* bottom_right_line is an averaging track of bottom right bbox point
		top_left_line = LineString([
			(first_bbox['x1'], first_bbox['y1']),
			(last_bbox['x1'], last_bbox['y1'])
		])
		bottom_right_line = LineString([
			(first_bbox['x2'], first_bbox['y2']),
			(last_bbox['x2'], last_bbox['y2'])
		])

		# Loop through all empty frames and interpolate them
		for step_id in range(1, len(interpolating_ids) + 1):
			processing_frame_id = first_id + step_id
			step = interpolation_step * step_id

			(tl_inter_x, tl_inter_y) = top_left_line.interpolate(step, normalized=True).coords[:][0]
			(br_inter_x, br_inter_y) = bottom_right_line.interpolate(step, normalized=True).coords[:][0]
			tl_inter_x, tl_inter_y = int(tl_inter_x), int(tl_inter_y)
			br_inter_x, br_inter_y = int(br_inter_x), int(br_inter_y)

			bboxes[processing_frame_id] = {
				"x1": tl_inter_x, "y1": tl_inter_y,
				"x2": br_inter_x, "y2": br_inter_y
			}

	return bboxes
