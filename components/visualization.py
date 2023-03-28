import os
import cv2
import numpy as np
from math import sqrt
from time import sleep
from shapely.geometry import LineString

from utils import read_json, create_folder, get_bbox_params
from cfg import cfg


joints_pairs = [
	['nose', 'left_eye'], ['nose', 'right_eye'],
	['left_eye', 'left_ear'], ['right_eye', 'right_ear'],
	['left_shoulder', 'right_shoulder'],
	['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
	['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
	['left_hip', 'right_hip'],
	['left_hip', 'left_knee'], ['left_knee', 'left_foot'],
	['right_hip', 'right_knee'], ['right_knee', 'right_foot'],
]


def get_caps_overlays(ballot_boxes, cap_polygons):
	"""
	Find:
		* ballot boxes stopping (nearby) zone (used for stopping detection)
		* ballot boxes lid zone
	"""
	int_coords = lambda x: np.array(x).round().astype(np.int32)
	accepted_box_distances = {}
	ballot_boxes_overlays = {}
	for ballot_box_id, ballot_box_dict in ballot_boxes.items():
		_, ballot_box_w, __ = get_bbox_params(
			ballot_box_dict['bbox']['x1'], ballot_box_dict['bbox']['y1'],
			ballot_box_dict['bbox']['x2'], ballot_box_dict['bbox']['y2'],
			out_type='int'
		)

		accepted_box_distances[ballot_box_id] = int(cfg.ballot_box_zone_k * ballot_box_w)

		if cfg.show_recognitions:
			# Get upscaled overlay of a lid zone (to visualize it on the frame)
			ballot_boxes_overlays[ballot_box_id] = [
				int_coords(cap_polygons[ballot_box_id].exterior.coords)
			]

	return accepted_box_distances, ballot_boxes_overlays


def draw_ballot_boxes(
		frame,
		ballot_boxes,
		ballot_boxes_overlays,
		cap_centroids_coords,
		accepted_box_distances,
		resized_width,
		resized_height,
		alpha=0.4,					# opacity of the lid zone
		draw_stop_zone=True			# whether to draw zone of stopping detection
):
	"""
	Visualize ballot boxes bboxes, lids and stopping zone on the frame
	"""
	for box_id in ballot_boxes.keys():
		overlay = frame.copy()
		cv2.fillPoly(overlay, ballot_boxes_overlays[box_id], color=(51, 255, 0))
		cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

		bbox_x1, bbox_y1, bbox_x2, bbox_y2 = \
			ballot_boxes[box_id]['bbox']['x1'], ballot_boxes[box_id]['bbox']['y1'], \
			ballot_boxes[box_id]['bbox']['x2'], ballot_boxes[box_id]['bbox']['y2']
		cap_c_x, cap_c_y = \
			int(ballot_boxes[box_id]['caps']['centroid_k']['x'] * resized_width), \
			int(ballot_boxes[box_id]['caps']['centroid_k']['y'] * resized_height)

		cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 136, 255), 1)
		cv2.circle(frame, (int(cap_c_x), int(cap_c_y)), 3, (0, 229, 15), -1)

		if ballot_boxes[box_id]['caps']['type'] == 'bbox':
			cap_bbox_x1, cap_bbox_y1, cap_bbox_x2, cap_bbox_y2 = \
				ballot_boxes[box_id]['caps']['upscaled_ort_bbox']['x1'], \
				ballot_boxes[box_id]['caps']['upscaled_ort_bbox']['y1'], \
				ballot_boxes[box_id]['caps']['upscaled_ort_bbox']['x2'],\
				ballot_boxes[box_id]['caps']['upscaled_ort_bbox']['y2']
			cv2.rectangle(
				frame,
				(cap_bbox_x1, cap_bbox_y1),
				(cap_bbox_x2, cap_bbox_y2),
				(0, 229, 15), 1
			)

		# Visualize stopping zone of the ballot box
		if draw_stop_zone:
			cv2.circle(
				frame,
				(cap_centroids_coords[box_id]['x'], cap_centroids_coords[box_id]['y']),
				accepted_box_distances[box_id], (0, 255, 0), 1
			)

	return frame


def draw_voter_bbox(
		frame, stopped_near_box, voter_bbox, voter_centroid, voter_id
):
	"""
	Visualize person bounding box.

	If a person has stopped near ballot box, yellow bbox will be drawn.
	Default color of bbox is white.
	"""

	text_colour = (255, 255, 255) if not stopped_near_box else (0, 213, 50)
	bbox_colour = (255, 255, 255) if not stopped_near_box else (21, 225, 255)

	cv2.rectangle(
		frame,
		(voter_bbox['x1'], voter_bbox['y1']),
		(voter_bbox['x2'], voter_bbox['y2']),
		bbox_colour, 1
	)
	cv2.circle(
		frame,
		(voter_centroid['x'], voter_centroid['y']),
		2, (21, 225, 255), -1
	)

	text = "# {}".format(voter_id)
	cv2.putText(
		frame, text, (voter_bbox['x1'], voter_bbox['y1'] - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_colour, 2
	)
	return frame


def draw_hands(
		img,
		joints,
		draw_shoulders=False,
		dot_size=1,
		left_hand_colour=(0, 0, 255),
		right_hand_colour=(255, 0, 255)
):
	"""
	Draw hands of a person. Following joints will be drawn:
		* elbows
		* wrists

	Right and left hands have different colors.
	"""

	for joint_name in joints:

		joint_x = joints[joint_name][0]
		joint_y = joints[joint_name][1]

		if joint_name in ['left_elbow', 'right_elbow']:
			dot_size = 3
		if joint_name in ['left_wrist', 'right_wrist']:
			dot_size = 2

		if joint_name in ['left_wrist', 'left_elbow']:
			cv2.circle(img, (int(joint_x), int(joint_y)), dot_size, left_hand_colour, -1)

		if joint_name in ['right_wrist', 'right_wrist']:
			cv2.circle(img, (int(joint_x), int(joint_y)), dot_size, right_hand_colour, -1)

	if 'left_elbow' in joints and 'left_wrist' in joints:
		cv2.line(
			img,
			(joints['left_elbow'][0], joints['left_elbow'][1]),
			(joints['left_wrist'][0], joints['left_wrist'][1]),
			left_hand_colour, 2
		)

	if 'right_elbow' in joints and 'right_wrist' in joints:
		cv2.line(
			img,
			(joints['right_elbow'][0], joints['right_elbow'][1]),
			(joints['right_wrist'][0], joints['right_wrist'][1]),
			right_hand_colour, 2
		)

	if draw_shoulders:
		if 'left_shoulder' in joints and 'right_shoulder' in joints:
			cv2.circle(
				img,
				(joints['left_shoulder'][0], joints['left_shoulder'][1]),
				2, left_hand_colour, -1
			)
			cv2.circle(
				img,
				(joints['right_shoulder'][0], joints['right_shoulder'][1]),
				2, right_hand_colour, -1
			)

	return img


def draw_skeleton(
		img,
		joints,
		draw_spine=True,		# whether to draw person's spine
		skeleton_colour=(21, 225, 255),
		dot_size=1,				# skeleton joint's point size
		alpha=0.3				# skeleton opacity
):
	"""
	Visualize skeleton of a person
	"""

	for joint_name in joints:
		joint_x = joints[joint_name][0]
		joint_y = joints[joint_name][1]

		overlay = img.copy()
		cv2.circle(overlay, (int(joint_x), int(joint_y)), dot_size, skeleton_colour, -1)
		cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

	for [pair_name_start, pair_name_end] in joints_pairs:

		if pair_name_start not in joints.keys() \
			and pair_name_end not in joints.keys():

			continue

		overlay = img.copy()
		cv2.line(
			overlay,
			(joints[pair_name_start][0], joints[pair_name_start][1]),
			(joints[pair_name_end][0], joints[pair_name_end][1]),
			skeleton_colour, 1
		)
		cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

	# Spine joints must be inside joints dict for drawing spine
	target_joints_inside = {
		'nose',
		'left_shoulder', 'right_shoulder',
		'left_hip', 'right_hip'
	}.issubset(set(joints.keys()))

	if draw_spine and target_joints_inside:

		# Find neck and sacrum coordinates
		shoulder_line = LineString([
			(joints['left_shoulder'][0], joints['left_shoulder'][1]),
			(joints['right_shoulder'][0], joints['right_shoulder'][1])
		])
		hip_line = LineString([
			(joints['left_hip'][0], joints['left_hip'][1]),
			(joints['right_hip'][0], joints['right_hip'][1])
		])

		neck = list(shoulder_line.centroid.coords)[0]
		sacrum = list(hip_line.centroid.coords)[0]

		overlay = img.copy()
		cv2.line(
			overlay,
			(int(neck[0]), int(neck[1])),
			(int(sacrum[0]), int(sacrum[1])),
			skeleton_colour, 1
		)
		cv2.line(
			overlay,
			(int(neck[0]), int(neck[1])),
			(int(joints['nose'][0]), int(joints['nose'][1])),
			skeleton_colour, 1
		)
		cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

	return img


def draw_labels(
		frame,
		frame_id,
		vote_info,
		draw_boxes=True,
		draw_caps=True,
		draw_voter_bboxes=True,
		draw_voter_poses=True,
		draw_text=True,
):
	"""
	Visualize sample data on a source frame.
	"""

	# Draw ballot boxes
	if draw_boxes:
		for box_id, box_bbox in vote_info['ballot_box_bbox'].items():
			text = "box #{}".format(box_id)
			cv2.putText(
				frame, text,
				(box_bbox['x1'], int(box_bbox['y1'] + (box_bbox['y2'] - box_bbox['y1']) * 0.66)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
			)
			cv2.rectangle(
				frame, (box_bbox['x1'], box_bbox['y1']), (box_bbox['x2'], box_bbox['y2']),
				(21, 150, 255), 1
			)

	# Draw ballot box lid
	if draw_caps:

		for box_id, cap_bbox in vote_info['cap_bbox'].items():
			cv2.rectangle(
				frame,
				(cap_bbox['x1'], cap_bbox['y1']),
				(cap_bbox['x2'], cap_bbox['y2']),
				(0, 229, 15), 1
			)

		for box_id, cap_centroid in vote_info['cap_centroid'].items():
			cv2.circle(
				frame,
				(cap_centroid['x'], cap_centroid['y']),
				3, (0, 229, 15), -1
			)

	# Draw person bounding box
	if draw_voter_bboxes:
		if str(frame_id) in vote_info['voter_bboxes'].keys():
			voter_bbox = vote_info['voter_bboxes'].get(str(frame_id), None)

			if voter_bbox:
				voter_x1, voter_y1, voter_x2, voter_y2 = \
					voter_bbox['x1'], voter_bbox['y1'], \
					voter_bbox['x2'], voter_bbox['y2']

				cv2.rectangle(
					frame,
					(voter_x1, voter_y1), (voter_x2, voter_y2),
					(228, 99, 0), 1
				)

	# Draw person's skeleton
	orientation_text = 'Person orientation: n/d, angle: n/d'
	if draw_voter_poses:
		if str(frame_id) in vote_info['joint_coords'].keys():
			joints = vote_info['joint_coords'].get(str(frame_id), None)

			frame = draw_skeleton(frame, joints)
			frame = draw_hands(frame, joints)

		# Draw direction of a person
		if str(frame_id) in vote_info['voter_orientation'].keys():
			orientation = vote_info['voter_orientation'].get(str(frame_id), None)
			if orientation:
				angle = orientation['angle']
				orientation_text = 'Person orientation: {}, angle: {:.2f}'.format(
					orientation['orientation'], angle
				)

				cv2.line(
					frame,
					(
						int(orientation['direction_line']['shoulders_mid_x1']),
						int(orientation['direction_line']['shoulders_mid_y1'])
					),
					(
						int(orientation['direction_line']['dir_x2']),
						int(orientation['direction_line']['dir_y2'])
					),
					(255, 255, 255), 2
				)

	if draw_text:
		vid_info = "Current frame ID: {}, voting start: {}, voting end: {}".format(
			frame_id, vote_info['voting_start_frame'], vote_info['voting_end_frame'])

		if 'voted_ballot_box_type' in vote_info.keys():
			ballot_box_type = 'ballot_box (id: 0)' \
				if vote_info['voted_ballot_box_type'] == 0 else 'koib (id: 1)'
		else:
			ballot_box_type = 'n/d'

		cv2.putText(
			frame, vid_info,
			(20, vote_info['video_height'] - 90),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 215), 2
		)
		cv2.putText(
			frame, orientation_text,
			(20, vote_info['video_height'] - 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 215), 2
		)
		cv2.putText(
			frame, 'Voted ballot box type: ' + ballot_box_type,
			(20, vote_info['video_height'] - 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 215), 2
		)

	return frame


def process_source_video(
		vid_path,
		json_path,
		delay_per_frame,
		vid_filename,
		labelled_vid_dir,
		show_window,
		draw_bg=True,
		draw_boxes=True,
		draw_caps=True,
		draw_voter_bboxes=True,
		draw_voter_poses=True,
		draw_text=True,
):
	"""
	Visualize labels on a single source video (without any visualized data).
	"""
	labelled_json = read_json(json_path)

	video_stream = cv2.VideoCapture(vid_path)
	got_frame, source_frame = video_stream.read()
	process = True if got_frame else False

	video_fps, source_height, source_width, \
		resized_height, resized_width, aspect_ratio = \
		get_video_stream_params(video_stream, source_frame, 640 * 480)

	output_video_path = os.path.join(labelled_vid_dir, vid_filename)

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	output_writer = cv2.VideoWriter(
		output_video_path, fourcc, video_fps,
		(resized_width, resized_height)
	)

	frame_id = 0
	while process:

		if draw_bg:
			# Draw labels on a source video
			frame_resized = cv2.resize(source_frame, (resized_width, resized_height))
		else:
			# Draw labels on a black background
			frame_resized = np.zeros((resized_height, resized_width, 3), np.uint8)

		labelled_frame = draw_labels(
			frame_resized, frame_id, labelled_json,
			draw_boxes=draw_boxes,
			draw_caps=draw_caps,
			draw_voter_bboxes=draw_voter_bboxes,
			draw_voter_poses=draw_voter_poses,
			draw_text=draw_text,
		)

		output_writer.write(labelled_frame)

		if show_window:
			cv2.imshow("frame", labelled_frame)
			sleep(delay_per_frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			process = False
		else:
			got_frame, source_frame = video_stream.read()
			process = True if got_frame else False
		frame_id += 1

	output_writer.release()
	cv2.destroyAllWindows()
	video_stream.release()


def get_video_stream_params(stream, src_frame, target_pixels):

	vid_fps = int(stream.get(cv2.CAP_PROP_FPS))
	(src_height, src_width) = src_frame.shape[:2]

	aspect_ratio = src_height / src_width
	resized_width = int(sqrt(target_pixels / aspect_ratio))
	resized_height = int(aspect_ratio * resized_width)

	return vid_fps, src_height, src_width, resized_height, resized_width, aspect_ratio


def main():

	# Visualize labels on all source videos:

	# Draw labels on a source video
	# (if set to False, black bg will be used)
	draw_bg = True

	# Show cv2 window with visualized (labelled) sample
	show_window = False
	# Slow down a little playback in cv2 (if show_window=True)
	delay_per_frame = 0.04

	# Output labelled videos directory
	labelled_vid_dir = 'path_to_output_dir'
	create_folder(labelled_vid_dir)

	# Source directories paths
	vid_dir = 'path_to_source_videos_dir'
	json_dir = 'path_to_jsons_dir'

	raw_videos = os.listdir(vid_dir)

	for vid_id, vid_filename in enumerate(raw_videos):

		print('{:.2f}%\t| {}'.format(vid_id / len(raw_videos) * 100, vid_filename))

		# Get original filename from json filename:
		# 6_40_1_vote37_62.json from 6_40_1_vote37_62.mp4
		# or 11_103_1_13.json from 11_103_1_13.mp4
		json_filename = '{}.json'.format(vid_filename.split('.')[0])

		# Get name without last element:
		# 6_40_1_vote37.json from 6_40_1_vote37_62.mp4
		# json_filename = '{}.json'.format('_'.join(vid_filename.split('.')[0].split('_')[:-1]))

		vid_path = os.path.join(vid_dir, vid_filename)
		json_path = os.path.join(json_dir, json_filename)

		if os.path.isfile(vid_path) and os.path.isfile(json_path):
			try:
				process_source_video(
					vid_path, json_path, delay_per_frame,
					vid_filename, labelled_vid_dir, show_window,
					draw_bg=draw_bg,
					draw_boxes=False,
					draw_caps=False,
					draw_voter_bboxes=True,
					draw_voter_poses=True,
					draw_text=True,
				)
			except:
				continue

		# sleep(1000)


if __name__ == '__main__':
	main()
