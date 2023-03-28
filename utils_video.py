import time

import cv2
from math import sqrt
from time import sleep

import mmcv
import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector
from mmcv.runner import load_checkpoint
from functools import partial
from shapely.geometry import Polygon

from cfg import cfg
from utils import chunk_list, get_bbox_params, upscale_poly_cap, upscale_bbox_cap, \
	coefficients_to_coordinates, get_bbox_from_coords
from components.visualization import draw_labels


def get_stream_params(stream, src_frame, target_pixels, find_fps=True):
	"""
	Find main video stream parameters:
		* frame width and height
		* aspect ratio
		* FPS
		* resized width and height with respect to target pixels value
	"""

	if find_fps:
		vid_fps = int(stream.get(cv2.CAP_PROP_FPS))
	(src_height, src_width) = src_frame.shape[:2]

	aspect_ratio = src_height / src_width
	resized_width = int(sqrt(target_pixels / aspect_ratio))
	resized_height = int(aspect_ratio * resized_width)

	return vid_fps if find_fps \
		else None, src_height, src_width, resized_height, resized_width, aspect_ratio


def frame_drop_needed(raw_video_fps, target_fps):
	"""
	Determine whether frames drop needed (for speeding up processing)
	"""
	frames_drop_needed = False
	if target_fps >= raw_video_fps:
		target_frames = list(range(0, raw_video_fps))
		target_fps_ratio = 1
		video_fps = raw_video_fps
	else:
		frame_ids = list(range(0, raw_video_fps))
		target_fps_ratio = raw_video_fps / target_fps
		chunked_list = chunk_list(frame_ids, target_fps)
		target_frames = [fr_id[0] for fr_id in chunked_list]
		video_fps = target_fps
		frames_drop_needed = True
	return frames_drop_needed, target_frames, target_fps_ratio, video_fps


def get_rec_zone_coords(bboxes, resized_width, resized_height):
	"""
	Find zone to process - ballot boxes will be centered,
	other area will be filled black
	"""
	contour_coords = []
	for box_id in bboxes.keys():
		box_bbox = bboxes[box_id]['bbox']

		_, box_width, box_height = get_bbox_params(
			box_bbox['x1'], box_bbox['y1'],
			box_bbox['x2'], box_bbox['y2'],
			out_type='int'
		)

		width_k = 0.7
		width_ratio = box_height / box_width
		height_ratio = 1 / width_ratio

		new_x1 = box_bbox['x1'] - int(box_width * width_ratio * (1 + width_k / 2))
		new_y1 = box_bbox['y1'] - int(box_height * height_ratio * 1.2 * (1 + width_k / 2))
		new_x2 = box_bbox['x2'] + int(box_width * width_ratio * (1 + width_k / 2))
		new_y2 = box_bbox['y2'] + int(box_height * height_ratio * 1.2 * (1 + width_k / 2))

		new_x1 = 0 if new_x1 < 0 else new_x1
		new_y1 = 0 if new_y1 < 0 else new_y1
		new_x2 = resized_width if new_x2 > resized_width else new_x2
		new_y2 = resized_height if new_y2 > resized_height else new_y2

		# Go the following way for making processing zone:
		# top left -> top right -> bottom right -> bottom left
		# * -> *
		#      |
		# * <- *
		contour_coords.append([new_x1, new_y1])
		contour_coords.append([new_x2, new_y1])
		contour_coords.append([new_x2, new_y2])
		contour_coords.append([new_x1, new_y2])

	all_x = [point[0] for point in contour_coords]
	all_y = [point[1] for point in contour_coords]
	min_x, min_y, max_x, max_y = min(all_x), min(all_y), max(all_x), max(all_y)
	contour_coords = [
		[min_x, min_y],
		[max_x, min_y],
		[max_x, max_y],
		[min_x, max_y]
	]
	return contour_coords


def vid_saver(
		output_frames,
		voter_dict=None,
		draw_recs=False,
		save_vid=False,
		output_video_path=None,
		show_window=False,			# whether to show cv2 window with saved video
		window_name='',
		fps=cfg.target_fps,
		width=640,
		height=480
):
	"""
	Save to the output video sample's frames and data.
	Visualizing data may be specified (if draw_recs=True)
	"""

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	output_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

	if save_vid or show_window:

		for prep_frame_id, prepared_frame in enumerate(output_frames):
			saving_frame = prepared_frame.copy()

			if draw_recs:
				saving_frame = draw_labels(saving_frame, prep_frame_id, voter_dict)

			if show_window:
				cv2.imshow(window_name, saving_frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				# Wait a little to make playback not too fast
				sleep(0.05)

			if save_vid:
				output_writer.write(saving_frame)

	if show_window:
		cv2.destroyWindow(window_name)

	if save_vid:
		output_writer.release()


def get_cap_polygons(ballot_boxes, width, height):
	"""
	Make shapely polygons of ballot boxes lid zones and compute lids centroids.

	Args:
		ballot_boxes: output dictionary from ballot boxes recognition module
		width: frame width
		height: frame height
	"""

	cap_centroids_coords = {}
	cap_polygons = {}
	for ballot_box_id, ballot_box_dict in ballot_boxes.items():

		cap_centroids_coords[ballot_box_id] = {
			'x': int(ballot_box_dict['caps']['centroid_k']['x'] * width),
			'y': int(ballot_box_dict['caps']['centroid_k']['y'] * height)
		}

		if ballot_box_dict['caps']['type'] == 'bbox':
			cap_polygons[ballot_box_id] = Polygon((
				(
					ballot_box_dict['caps']['upscaled_ort_bbox']['x1'],
				 	ballot_box_dict['caps']['upscaled_ort_bbox']['y1']
				),
				(
					ballot_box_dict['caps']['upscaled_ort_bbox']['x1'],
				 	ballot_box_dict['caps']['upscaled_ort_bbox']['y2']
				),
				(
					ballot_box_dict['caps']['upscaled_ort_bbox']['x2'],
				 	ballot_box_dict['caps']['upscaled_ort_bbox']['y2']
				),
				(
					ballot_box_dict['caps']['upscaled_ort_bbox']['x2'],
					ballot_box_dict['caps']['upscaled_ort_bbox']['y1']
				)
			))

		elif ballot_box_dict['caps']['type'] == 'poly':
			cap_polygons[ballot_box_id] = Polygon((
				(
					ballot_box_dict['caps']['upscaled_rot_bbox']['top_left']['x'],
					ballot_box_dict['caps']['upscaled_rot_bbox']['top_left']['y']
				),
				(
					ballot_box_dict['caps']['upscaled_rot_bbox']['bottom_left']['x'],
					ballot_box_dict['caps']['upscaled_rot_bbox']['bottom_left']['y']
				),
				(
					ballot_box_dict['caps']['upscaled_rot_bbox']['bottom_right']['x'],
					ballot_box_dict['caps']['upscaled_rot_bbox']['bottom_right']['y']
				),
				(
					ballot_box_dict['caps']['upscaled_rot_bbox']['top_right']['x'],
					ballot_box_dict['caps']['upscaled_rot_bbox']['top_right']['y']
				)
			))

	return cap_polygons, cap_centroids_coords


def upscale_boxes(ballot_boxes, resized_width, resized_height):
	"""
	Upscale ballot box lid zone
	"""
	updated_ballot_boxes = ballot_boxes
	for ballot_box_id, ballot_box_dict in ballot_boxes.items():

		# Upscale polygon (rotated rectangle)
		if ballot_box_dict['caps']['type'] == 'poly':

			# Convert dict to list
			poly_points = [
				[dot['x'], dot['y']] for dot_name, dot
				in ballot_box_dict['caps']['rot_bbox'].items()
			]
			bbox_points = [
				[
					ballot_box_dict['caps']['ort_bbox']['x1'],
					ballot_box_dict['caps']['ort_bbox']['y1']
				],
				[
					ballot_box_dict['caps']['ort_bbox']['x2'],
					ballot_box_dict['caps']['ort_bbox']['y2']
				]
			]

			updated_ballot_boxes[ballot_box_id]['caps']['upscaled_rot_bbox'] = \
				upscale_poly_cap(poly_points, bbox_points)

		# Upscale bbox (orthogonal rectangle)
		updated_ballot_boxes[ballot_box_id]['caps']['upscaled_ort_bbox'] = \
			upscale_bbox_cap(
				ballot_box_dict['caps']['ort_bbox'],
				resized_width, resized_height
			)

	return updated_ballot_boxes


def convert_to_coordinates(
		ballot_boxes: dict,
		resized_width: int,
		resized_height: int
):
	"""
	Convert all ballot boxes coefficients (normalized coordinates) to coordinates

	Args:
		ballot_boxes: ballot boxes dictionary
		resized_width: frame width
		resized_height: frame height

	Returns:
		Updated ballot boxes dict
	"""
	converted_ballot_boxes = ballot_boxes
	for ballot_box_id, ballot_box_dict in ballot_boxes.items():

		ballot_box_bbox = coefficients_to_coordinates(
			ballot_box_dict['bbox_k'], resized_width, resized_height
		)
		ballot_box_centroid = coefficients_to_coordinates(
			ballot_box_dict['centroid_k'], resized_width, resized_height
		)

		cap_rot_bbox_top_right = coefficients_to_coordinates(
			ballot_box_dict['caps']['rot_bbox_k']['top_right'],
			resized_width, resized_height
		) if ballot_box_dict['caps']['rot_bbox_k'] else {}

		cap_rot_bbox_top_left = coefficients_to_coordinates(
			ballot_box_dict['caps']['rot_bbox_k']['top_left'],
			resized_width, resized_height
		) if ballot_box_dict['caps']['rot_bbox_k'] else {}

		cap_rot_bbox_bottom_left = coefficients_to_coordinates(
			ballot_box_dict['caps']['rot_bbox_k']['bottom_left'],
			resized_width, resized_height
		) if ballot_box_dict['caps']['rot_bbox_k'] else {}

		cap_rot_bbox_bottom_right = coefficients_to_coordinates(
			ballot_box_dict['caps']['rot_bbox_k']['bottom_right'],
			resized_width, resized_height
		) if ballot_box_dict['caps']['rot_bbox_k'] else {}

		converted_ballot_boxes[ballot_box_id]['bbox'] = ballot_box_bbox
		converted_ballot_boxes[ballot_box_id]['centroid'] = ballot_box_centroid

		if cap_rot_bbox_top_right and cap_rot_bbox_top_left \
			and cap_rot_bbox_bottom_left and cap_rot_bbox_bottom_right:

			converted_ballot_boxes[ballot_box_id]['caps']['rot_bbox'] = {
				'top_right': cap_rot_bbox_top_right,
				'top_left': cap_rot_bbox_top_left,
				'bottom_left': cap_rot_bbox_bottom_left,
				'bottom_right': cap_rot_bbox_bottom_right,
			}

		cap_centroid = coefficients_to_coordinates(
			ballot_box_dict['caps']['centroid_k'],
			resized_width, resized_height
		) if ballot_box_dict['caps']['centroid_k'] else {}
		converted_ballot_boxes[ballot_box_id]['caps']['centroid'] = cap_centroid

		# If ort_bbox has not been received from API, find it from polygon
		if not ballot_box_dict['caps']['ort_bbox_k']:
			poly_points = [
				[dot['x'], dot['y']] for dot_name, dot in
				converted_ballot_boxes[ballot_box_id]['caps']['rot_bbox'].items()
			]
			cap_bbox = get_bbox_from_coords(poly_points)
			cap_ort_bbox = {
				'x1': cap_bbox[0], 'y1': cap_bbox[1],
				'x2': cap_bbox[2], 'y2': cap_bbox[3]
			}
		else:
			cap_ort_bbox = coefficients_to_coordinates(
				ballot_box_dict['caps']['ort_bbox_k'],
				resized_width, resized_height
			)

		converted_ballot_boxes[ballot_box_id]['caps']['ort_bbox'] = cap_ort_bbox

	return converted_ballot_boxes


def generate_inputs_and_wrap_model(
		config_path, checkpoint_path, input_config, cfg_options=None
):
	"""Prepare sample input and wrap model for ONNX export.

	The ONNX export API only accept args, and all inputs should be
	torch.Tensor or corresponding types (such as tuple of tensor).
	So we should call this function before exporting. This function will:

	1. generate corresponding inputs which are used to execute the model.
	2. Wrap the model's forward function.

	For example, the MMDet models' forward function has a parameter
	``return_loss:bool``. As we want to set it as False while export API
	supports neither bool type or kwargs. So we have to replace the forward
	method like ``model.forward = partial(model.forward, return_loss=False)``.

	Args:
		config_path (str): the OpenMMLab config for the model we want to
			export to ONNX
		checkpoint_path (str): Path to the corresponding checkpoint
		input_config (dict): the exactly data in this dict depends on the
			framework. For MMSeg, we can just declare the input shape,
			and generate the dummy data accordingly. However, for MMDet,
			we may pass the real img path, or the NMS will return None
			as there is no legal bbox.

	Returns:
		tuple: (model, tensor_data) wrapped model which can be called by
			``model(*tensor_data)`` and a list of inputs which are used to
			execute the model while exporting.
	"""

	model = build_model_from_cfg(
		config_path, checkpoint_path, cfg_options=cfg_options)
	one_img, one_meta = preprocess_example_input(input_config)
	tensor_data = [one_img]
	model.forward = partial(
		model.forward, img_metas=[[one_meta]], return_loss=False)

	# pytorch has some bug in pytorch1.3, we have to fix it
	# by replacing these existing op
	opset_version = 11
	# put the import within the function thus it will not cause import error
	# when not using this function
	try:
		from mmcv.onnx.symbolic import register_extra_symbolics
	except ModuleNotFoundError:
		raise NotImplementedError('please update mmcv to version>=v1.0.4')
	register_extra_symbolics(opset_version)

	return model, tensor_data


def build_model_from_cfg(config_path, checkpoint_path, cfg_options=None):
	"""Build a model from config and load the given checkpoint.

	Args:
		config_path (str): the OpenMMLab config for the model we want to
			export to ONNX
		checkpoint_path (str): Path to the corresponding checkpoint

	Returns:
		torch.nn.Module: the built model
	"""
	from mmdet.models import build_detector

	device = 'cpu'
	cfg = mmcv.Config.fromfile(config_path)
	if cfg_options is not None:
		cfg.merge_from_dict(cfg_options)
	# import modules from string list.
	if cfg.get('custom_imports', None):
		from mmcv.utils import import_modules_from_strings
		import_modules_from_strings(**cfg['custom_imports'])
	# set cudnn_benchmark
	if cfg.get('cudnn_benchmark', False):
		torch.backends.cudnn.benchmark = True
	cfg.model.pretrained = None
	cfg.data.test.test_mode = True

	# build the model
	cfg.model.train_cfg = None
	model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
	checkpoint = load_checkpoint(
		model, checkpoint_path, map_location=device, logger='silent'
	)
	if 'CLASSES' in checkpoint.get('meta', {}):
		model.CLASSES = checkpoint['meta']['CLASSES']
	else:
		from mmdet.datasets import DATASETS
		dataset = DATASETS.get(cfg.data.test['type'])
		assert (dataset is not None)
		model.CLASSES = dataset.CLASSES
	model.cpu().eval()
	return model


def preprocess_example_input(input_config):
	"""Prepare an example input image for ``generate_inputs_and_wrap_model``.

	Args:
		input_config (dict): customized config describing the example input.

	Returns:
		tuple: (one_img, one_meta), tensor of the example input image and \
			meta information for the example input image.

	Examples:
		# >>> from mmdet.core.export import preprocess_example_input
		# >>> input_config = {
		# >>>         'input_shape': (1,3,224,224),
		# >>>         'input_path': 'demo/demo.jpg',
		# >>>         'normalize_cfg': {
		# >>>             'mean': (123.675, 116.28, 103.53),
		# >>>             'std': (58.395, 57.12, 57.375)
		# >>>             }
		# >>>         }
		# >>> one_img, one_meta = preprocess_example_input(input_config)
		# >>> print(one_img.shape)
		# torch.Size([1, 3, 224, 224])
		# >>> print(one_meta)
		{'img_shape': (224, 224, 3),
		'ori_shape': (224, 224, 3),
		'pad_shape': (224, 224, 3),
		'filename': '<demo>.png',
		'scale_factor': 1.0,
		'flip': False}
	"""
	input_path = input_config['input_path']
	input_shape = input_config['input_shape']
	one_img = mmcv.imread(input_path)
	one_img = mmcv.imresize(one_img, input_shape[2:][::-1])
	show_img = one_img.copy()
	if 'normalize_cfg' in input_config.keys():
		normalize_cfg = input_config['normalize_cfg']
		mean = np.array(normalize_cfg['mean'], dtype=np.float32)
		std = np.array(normalize_cfg['std'], dtype=np.float32)
		to_rgb = normalize_cfg.get('to_rgb', True)
		one_img = mmcv.imnormalize(one_img, mean, std, to_rgb=to_rgb)
	one_img = one_img.transpose(2, 0, 1)
	one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(True)
	(_, C, H, W) = input_shape
	one_meta = {
		'img_shape': (H, W, C),
		'ori_shape': (H, W, C),
		'pad_shape': (H, W, C),
		'filename': '<demo>.png',
		'scale_factor': np.ones(4, dtype=np.float32),
		'flip': False,
		'show_img': show_img,
		'flip_direction': None
	}

	return one_img, one_meta


def mmcv_process_video(
		config_name, weights, video_path, out_path, verbose,
		min_thresh=0.4, gpu_id=None
):

	convert_to_onnx = False
	# device = f'cuda:{gpu_id}' if gpu_id else 'cuda:0'
	device = 'cuda:{}'.format(gpu_id) if gpu_id else 'cuda:0'

	raw_recognitions = []
	raw_masks = []
	if convert_to_onnx:
		input_config = {
			'input_shape': (1, 3, 480, 640),
			'input_path': 'QueryInst/demo/demo.jpg',
		}

		model, _ = generate_inputs_and_wrap_model(
			config_name, weights, input_config, cfg_options=None
		)
		model.cuda(device)
	else:
		model = init_detector(config_name, weights, device=device)
	video = cv2.VideoCapture(video_path)
	counter = 0
	if verbose:
		prog_bar = mmcv.ProgressBar(round(video.get(cv2.CAP_PROP_FRAME_COUNT)))

	if out_path:
		ret, img = video.read()
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		vid_fps = int(video.get(cv2.CAP_PROP_FPS))
		writer = cv2.VideoWriter(
			out_path, fourcc, vid_fps, (img.shape[1], img.shape[0])
		)

	while True:
		ret, img = video.read()
		if ret:
			if convert_to_onnx:
				one_img = img.copy()
				input_shape = [1, 3, one_img.shape[0], one_img.shape[1]]
				one_img = mmcv.imresize(one_img, tuple(input_shape[2:][::-1]))
				one_img = one_img.transpose(2, 0, 1)
				one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(True)
				tensor_data = [one_img.cuda()]
				result = model(tensor_data)[0]
			else:
				result = inference_detector(model, img)
			start_time = time.strftime("%Y-%m-%d %H:%M:%S.%s")
			frame_height, frame_width, = img.shape[0:2]
			objects = {}
			objects_mask = {}

			if isinstance(result, tuple):
				bbox_result, segm_result = result
				if isinstance(segm_result, tuple):
					segm_result = segm_result[0]  # ms rcnn
			else:
				bbox_result, segm_result = result, None
			bboxes = np.vstack(bbox_result)
			labels = [
				np.full(bbox.shape[0], i, dtype=np.int32)
				for i, bbox in enumerate(bbox_result)
			]
			labels = np.concatenate(labels)
			# draw segmentation masks
			segms = None
			if segm_result is not None and len(labels) > 0:  # non empty
				segms = mmcv.concat_list(segm_result)
				if isinstance(segms[0], torch.Tensor):
					segms = torch.stack(segms, dim=0).detach().cpu().numpy()
				else:
					segms = np.stack(segms, axis=0)

			if min_thresh > 0:
				assert bboxes.shape[1] == 5
				scores = bboxes[:, -1]
				inds = scores > min_thresh
				bboxes = bboxes[inds, :]
				if segms is not None:
					segms = segms[inds, ...]

				for i, (segm, bbox) in enumerate(zip(segms, bboxes)):

					contours, _ = cv2.findContours(
						segm.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
					)

					polygons = []
					for object in contours:
						coords = []
						for point in object:
							coords.append([int(point[0][0]), int(point[0][1])])
						polygons.append(coords)

					objects_mask[i] = {}

					count_true = 0
					for res in polygons:
						objects_mask[i][count_true] = res
						count_true += 1

					objects[i] = {
						'class_name': 'b',
						'conf': str(bbox[4]),
						'coords': [
							[str(int(bbox[0])), str(int(bbox[1]))],
							[str(int(bbox[2])), str(int(bbox[3]))]
						]
					}

				raw_masks.append({
					'rec_time': start_time,
					'objects': objects_mask,
					'frame_id': counter
				})
				raw_recognitions.append({
					'rec_time': start_time,
					'frame_width': str(frame_width),
					'frame_height': str(frame_height),
					'objects': objects,
					'frame_id': counter
				})

			rec_img = model.show_result(
				img, result,
				show=False, font_size=7,
				bbox_color=(0, 255, 0),
				thickness=1, score_thr=min_thresh
			)

			counter += 1

			if out_path:
				writer.write(rec_img)
			if verbose:
				prog_bar.update()
		else:
			break

	video.release()
	if out_path:
		writer.release()

	return raw_recognitions, raw_masks


def init_mmcv_model(config_name, weights, gpu_id=None):
	convert_to_onnx = False
	# device = f'cuda:{gpu_id}' if gpu_id else 'cuda:0'
	device = 'cuda:{}'.format(gpu_id) if gpu_id else 'cuda:0'
	if convert_to_onnx:
		input_config = {
			'input_shape': (1, 3, 480, 640),
			'input_path': 'QueryInst/demo/demo.jpg',
		}

		model, _ = generate_inputs_and_wrap_model(
			config_name, weights, input_config, cfg_options=None
		)
		model.cuda(device)
	else:
		model = init_detector(config_name, weights, device=device)

	return model


def mmcv_process_image(model, img, min_thresh=0.3, out_path=None):
	convert_to_onnx = False

	if convert_to_onnx:
		input_shape = [1, 3, img.shape[0], img.shape[1]]
		img = mmcv.imresize(img, tuple(input_shape[2:][::-1]))
		img = img.transpose(2, 0, 1)
		img = torch.from_numpy(img).unsqueeze(0).float().requires_grad_(True)
		result = model(img)
	else:
		result = inference_detector(model, img)

	raw_recognitions = []
	raw_masks = []
	start_time = time.strftime("%Y-%m-%d %H:%M:%S.%s")
	frame_height, frame_width, = img.shape[0:2]
	if isinstance(result, tuple):
		bbox_result, segm_result = result
		if isinstance(segm_result, tuple):
			segm_result = segm_result[0]  # ms rcnn
	else:
		bbox_result, segm_result = result, None
	bboxes = np.vstack(bbox_result)
	labels = [
		np.full(bbox.shape[0], i, dtype=np.int32)
		for i, bbox in enumerate(bbox_result)
	]
	labels = np.concatenate(labels)

	labels_name = ['ballot_box', 'koib']
	# draw segmentation masks
	segms = None
	if segm_result is not None:  # non empty
		segms = mmcv.concat_list(segm_result)
		if isinstance(segms[0], torch.Tensor):
			segms = torch.stack(segms, dim=0).detach().cpu().numpy()
		else:
			segms = np.stack(segms, axis=0)

	scores = bboxes[:, -1]
	inds = scores > min_thresh
	bboxes = bboxes[inds, :]
	labels = labels[inds]
	if segms is not None:
		segms = segms[inds, ...]

	objects = {}
	objects_mask = {}

	for i, (segm, bbox, label) in enumerate(zip(segms, bboxes, labels)):
		mask = np.reshape(
			list(int(y) for x in segm for y in x), (segm.shape[0], segm.shape[1])
		)
		contours, _ = cv2.findContours(
			mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
		)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
		contours = cv2.convexHull(contours)

		coords = []
		for point in contours:
			coords.append([int(point[0][0]), int(point[0][1])])

		objects_mask[i] = coords
		objects[i] = {
			'class_name': labels_name[label],
			'conf': str(bbox[4]),
			'coords': [
				[str(int(bbox[0])), str(int(bbox[1]))],
				[str(int(bbox[2])), str(int(bbox[3]))]
			]
		}

	raw_masks.append({
		'rec_time': start_time,
		'objects': objects_mask,
		'frame_id': 0
	})
	raw_recognitions.append({
		'rec_time': start_time,
		'frame_width': str(frame_width),
		'frame_height': str(frame_height),
		'objects': objects,
		'frame_id': 0
	})

	image = model.show_result(
		img, result, show=False, font_size=7,
		bbox_color=(0, 255, 0), thickness=1,
		score_thr=min_thresh
	)
	if out_path:
		# cv2.imwrite(f'{out_path}.jpg', image)
		cv2.imwrite('{}.jpg'.format(out_path), image)

	return image, raw_recognitions, raw_masks
