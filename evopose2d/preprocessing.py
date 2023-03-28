import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf

from tensorflow.python.keras.layers.preprocessing import image_preprocessing as image_ops
import numpy as np


def prepare_evo_input(
		bboxes,
		prepared_frames,
		min_frame_id,
		max_frame_id,
		batch_size=32,
		pose_input_shape=(384, 288, 3)
):

	# Get list of source frame_ids to predict pose on them
	saving_frame_ids = list(range(min_frame_id, max_frame_id + 1))
	old_frames_list = list(prepared_frames.keys())

	max_bbox_width, max_bbox_height = 0, 0
	target_bboxes, target_images = {}, {}
	for old_frame_id in saving_frame_ids:

		if old_frame_id in old_frames_list:

			bbox = bboxes[old_frame_id]
			target_bboxes[old_frame_id] = bbox
			target_images[old_frame_id] = prepared_frames[old_frame_id]

			# Find max width and height in batch
			bbox_width, bbox_height = bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1']
			max_bbox_width = bbox_width if bbox_width > max_bbox_width else max_bbox_width
			max_bbox_height = bbox_height if bbox_height > max_bbox_height else max_bbox_height

	# Crop person bboxes from source images
	cropped_frames, cropped_bboxes, origins, old_frame_ids = crop_person_images(
		target_images, target_bboxes, max_bbox_width, max_bbox_height
	)

	# Preprocess stacked images
	bbox_batch = np.array(cropped_bboxes)
	img_batch = np.stack(cropped_frames)
	images_to_preprocess = len(target_images.keys())
	normalized_images, Ms = preproccess_images(img_batch, pose_input_shape, bbox_batch, images_to_preprocess)

	# Split all images to batches.
	# Each batch is filling up to max batch_size,
	# last element has remaining images
	batches_images = [normalized_images[i:i + batch_size] for i in range(0, normalized_images.shape[0], batch_size)]
	batches_Ms = [Ms[i:i + batch_size] for i in range(0, Ms.shape[0], batch_size)]

	return batches_images, batches_Ms, origins, old_frame_ids


def preproccess_images(
		img_batch, input_shape, bbox_batch, batch_size,
		means=(0.485, 0.456, 0.406),
		stds=(0.229, 0.224, 0.225)
):

	center_batch_x = bbox_batch[:, 0] + bbox_batch[:, 2] / 2.
	center_batch_y = bbox_batch[:, 1] + bbox_batch[:, 3] / 2.
	center_batch = np.stack([center_batch_x, center_batch_y], axis=-1)
	center_batch = tf.cast(center_batch, tf.float32)

	aspect_ratio = input_shape[1] / input_shape[0]
	scaling_mask = bbox_batch[:, 2] > bbox_batch[:, 3] / aspect_ratio
	bbox_batch[scaling_mask, 3] = bbox_batch[scaling_mask, 2] / aspect_ratio

	scale_batch = (bbox_batch[:, 3] * 1.25) / input_shape[0]

	img_batch = tf.cast(img_batch, tf.float32)
	img_batch /= 255.
	img_batch -= [[means]]
	img_batch /= [[stds]]

	tx = center_batch[:, 0] - input_shape[1] * scale_batch / 2
	ty = center_batch[:, 1] - input_shape[0] * scale_batch / 2

	zero_vec = np.zeros((batch_size))
	transform = np.stack([
		scale_batch, zero_vec, tx,
		zero_vec, scale_batch, ty,
		zero_vec, zero_vec
	], axis=-1)

	preproc_batch_img = image_ops.transform(
		img_batch, transform, fill_mode='constant',
		output_shape=[input_shape[0], input_shape[1]]
	)

	alpha = 1 / scale_batch * 1
	beta = 1 / scale_batch * 0

	rx_xy = (- alpha + 1) * center_batch[:, 0] - beta * center_batch[:, 1]
	ry_xy = beta * center_batch[:, 0] + (- alpha + 1) * center_batch[:, 1]

	transform_xy = np.zeros((batch_size, 2, 2))
	transform_xy[:, 0, 0] = alpha
	transform_xy[:, 0, 1] = beta
	transform_xy[:, 1, 0] = -beta
	transform_xy[:, 1, 1] = alpha

	tx_xy = center_batch[:, 0] - input_shape[1] / 2
	ty_xy = center_batch[:, 1] - input_shape[0] / 2

	tr = np.zeros((batch_size, 2, 1,))
	tr[:, 0, 0] = rx_xy - tx_xy
	tr[:, 1, 0] = ry_xy - ty_xy

	M = tf.concat([transform_xy, tr], axis=-1)
	M = np.array(M)

	return preproc_batch_img, M


def crop_person_images(frames, bboxes, max_bbox_width, max_bbox_height):

	cropped_frames, cropped_bboxes = [], []
	old_frame_ids = []
	origins = []
	for frame_id, (old_fr_id, bbox) in enumerate(bboxes.items()):

		frame = frames[old_fr_id]

		# Verify all coordinates to have positive values
		x1 = 0 if bbox['x1'] < 0 else bbox['x1']
		y1 = 0 if bbox['y1'] < 0 else bbox['y1']

		# Crop person's bbox
		cropped_image = frame[y1:bbox['y2'], x1:bbox['x2']]
		cropped_bbox = [0, 0, bbox['x2'] - x1, bbox['y2'] - y1]

		# Fill with zeros empty pixels (black color).
		# All batch images has the same size: (max_bbox_height, max_bbox_width)
		(cr_height, cr_width) = cropped_image.shape[:2]
		shift_width = max_bbox_width - cr_width
		shift_height = max_bbox_height - cr_height

		shifted_size = (cr_height + shift_height, cr_width + shift_width, 3)
		shifted_image = np.zeros(shifted_size, dtype=np.uint8)
		shifted_image[0:cr_height, 0:cr_width] = cropped_image

		cropped_frames.append(shifted_image)
		cropped_bboxes.append(cropped_bbox)

		old_frame_ids.append(old_fr_id)
		origins.append([bbox['x1'], bbox['y1']])
	return cropped_frames, cropped_bboxes, np.array(origins), old_frame_ids


def get_joints_dict(joints_list):
	return {
		'nose': joints_list[0],
		'left_eye': joints_list[1], 'right_eye': joints_list[2],
		'left_ear': joints_list[3], 'right_ear': joints_list[4],
		'left_shoulder': joints_list[5], 'right_shoulder': joints_list[6],
		'left_elbow': joints_list[7], 'right_elbow': joints_list[8],
		'left_wrist': joints_list[9], 'right_wrist': joints_list[10],
		'left_hip': joints_list[11], 'right_hip': joints_list[12],
		'left_knee': joints_list[13], 'right_knee': joints_list[14],
		'left_foot': joints_list[15], 'right_foot': joints_list[16]
	}


def get_joints_confs(coefs_list):
	return {
		'nose': coefs_list[0],
		'left_eye': coefs_list[1], 'right_eye': coefs_list[2],
		'left_ear': coefs_list[3], 'right_ear': coefs_list[4],
		'left_shoulder': coefs_list[5], 'right_shoulder': coefs_list[6],
		'left_elbow': coefs_list[7], 'right_elbow': coefs_list[8],
		'left_wrist': coefs_list[9], 'right_wrist': coefs_list[10],
		'left_hip': coefs_list[11], 'right_hip': coefs_list[12],
		'left_knee': coefs_list[13], 'right_knee': coefs_list[14],
		'left_foot': coefs_list[15], 'right_foot': coefs_list[16]
	}