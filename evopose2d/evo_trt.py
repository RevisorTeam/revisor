import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Comment 'autoinit' import if you are launching revisor.py
# Uncomment 'autoinit' import if you are launching evo_trt.py
# import pycuda.autoinit

import pycuda.driver as cuda

import numpy as np
import cv2
import tensorrt as trt
import math

import tensorflow as tf
from tensorflow.python.keras.layers.preprocessing import image_preprocessing as image_ops


class HostDeviceMem(object):
	"""Simple helper data class that's a little nicer to use than a 2-tuple."""
	def __init__(self, host_mem, device_mem):
		self.host = host_mem
		self.device = device_mem

	def __str__(self):
		return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

	def __repr__(self):
		return self.__str__()


def allocate_buffers(engine):
	"""Allocates all host/device in/out buffers required for an engine."""
	inputs = []
	outputs = []
	bindings = []
	memories = []
	output_idx = 0
	stream = cuda.Stream()
	assert len(engine) == 2
	for binding in engine:
		binding_dims = engine.get_binding_shape(binding)
		size = trt.volume(binding_dims)
		dtype = trt.nptype(engine.get_binding_dtype(binding))
		# Allocate host and device buffers
		host_mem = cuda.pagelocked_empty(size, dtype)
		device_mem = cuda.mem_alloc(host_mem.nbytes)
		memories.append(device_mem)
		# Append the device buffer to device bindings.
		bindings.append(int(device_mem))
		# Append to the appropriate list.
		if engine.binding_is_input(binding):
			inputs.append(HostDeviceMem(host_mem, device_mem))
		else:
			outputs.append(HostDeviceMem(host_mem, device_mem))
			output_idx += 1
	return inputs, outputs, bindings, stream, memories


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
	"""do_inference (for TensorRT 6.x or lower)

	This function is generalized for multiple inputs/outputs.
	Inputs and outputs are expected to be lists of HostDeviceMem objects.
	"""
	# Transfer input data to the GPU.
	[cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
	# Run inference.
	context.execute_async(
		batch_size=batch_size,
		bindings=bindings,
		stream_handle=stream.handle
	)
	# Transfer predictions back from the GPU.
	[cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
	# Synchronize the stream
	stream.synchronize()
	# Return only the host outputs.
	return [out.host for out in outputs]


def do_inference_v2(context, bindings, inputs, outputs, stream):
	"""do_inference_v2 (for TensorRT 7.0+)

	This function is generalized for multiple inputs/outputs for full
	dimension networks.
	Inputs and outputs are expected to be lists of HostDeviceMem objects.
	"""
	# Transfer input data to the GPU.
	[cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
	# Run inference.
	context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
	# Transfer predictions back from the GPU.
	[cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
	# Synchronize the stream
	stream.synchronize()
	# Return only the host outputs.
	return [out.host for out in outputs]


def transform(img, scale, angle, bbox_center, output_shape):
	tx = bbox_center[0] - output_shape[1] * scale / 2
	ty = bbox_center[1] - output_shape[0] * scale / 2

	# for offsetting translations caused by rotation:
	# https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
	rx = (1 - tf.cos(angle)) * output_shape[1] * scale / 2 - tf.sin(angle) * output_shape[0] * scale / 2
	ry = tf.sin(angle) * output_shape[1] * scale / 2 + (1 - tf.cos(angle)) * output_shape[0] * scale / 2

	transform = [
		scale * tf.cos(angle), scale * tf.sin(angle), rx + tx,
		-scale * tf.sin(angle), scale * tf.cos(angle), ry + ty,
		0., 0.
	]

	img = image_ops.transform(
		tf.expand_dims(img, axis=0),
		tf.expand_dims(transform, axis=0),
		fill_mode='constant',
		output_shape=output_shape[:2]
	)
	img = tf.squeeze(img)

	# transform for keypoints
	alpha = 1 / scale * tf.cos(-angle)
	beta = 1 / scale * tf.sin(-angle)

	rx_xy = (1 - alpha) * bbox_center[0] - beta * bbox_center[1]
	ry_xy = beta * bbox_center[0] + (1 - alpha) * bbox_center[1]

	transform_xy = [[alpha, beta],
					[-beta, alpha]]

	tx_xy = bbox_center[0] - output_shape[1] / 2
	ty_xy = bbox_center[1] - output_shape[0] / 2

	M = tf.concat([transform_xy, [[rx_xy - tx_xy], [ry_xy - ty_xy]]], axis=1)
	return img, M


def get_preds(hms, Ms, input_shape, output_shape, images_in_batch=None):

	images_number = hms.shape[0] if images_in_batch is None else images_in_batch

	preds = np.zeros((images_number, output_shape[-1], 3))
	for i in range(preds.shape[0]):
		for j in range(preds.shape[1]):
			hm = hms[i, :, :, j]
			idx = hm.argmax()
			y, x = np.unravel_index(idx, hm.shape)
			px = int(math.floor(x + 0.5))
			py = int(math.floor(y + 0.5))
			if 1 < px < output_shape[1] - 1 and 1 < py < output_shape[0] - 1:
				diff = np.array([hm[py][px + 1] - hm[py][px - 1],
								 hm[py + 1][px] - hm[py - 1][px]])
				diff = np.sign(diff)
				x += diff[0] * 0.25
				y += diff[1] * 0.25
			preds[i, j, :2] = [x * input_shape[1] / output_shape[1],
							   y * input_shape[0] / output_shape[0]]
			preds[i, j, -1] = hm.max() / 255

	# use inverse transform to map kp back to original image
	for j in range(preds.shape[0]):
		# print(Ms[j])
		M_inv = cv2.invertAffineTransform(Ms[j])
		preds[j, :, :2] = np.matmul(M_inv[:, :2], preds[j, :, :2].T).T + M_inv[:, 2].T
	return preds


def preprocess_single_image(
		img, bbox, input_shape,
		angle=0.0,
		means=(0.485, 0.456, 0.406),
		stds=(0.229, 0.224, 0.225)
):

	img = np.array(img)[:, :, 0:3]
	img = tf.cast(img, tf.float32)
	img /= 255.
	img -= [[means]]
	img /= [[stds]]

	x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
	center = [x + w / 2., y + h / 2.]
	center = tf.cast(tf.stack(center), tf.float32)
	aspect_ratio = input_shape[1] / input_shape[0]
	if w > aspect_ratio * h:
		h = w / aspect_ratio
	scale = (h * 1.25) / input_shape[0]
	img, M = transform(img, scale, angle, center, input_shape[:2])

	M = np.expand_dims(np.array(M), axis=0)
	img_preprocessed = np.expand_dims(img, axis=0)

	return img_preprocessed, M


class TrtEVO(object):

	def _load_engine(self):
		with open(self.model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
			return runtime.deserialize_cuda_engine(f.read())

	def __init__(self, model_path, input_shape, output_shape, batch_size=1,
				 cuda_ctx=None):
		"""Initialize TensorRT plugins, engine and context."""
		self.model_path = model_path
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.batch_size = batch_size

		self.reshaping_dims = (self.batch_size,) + tuple(self.output_shape)

		self.cuda_ctx = cuda_ctx
		if self.cuda_ctx:
			self.cuda_ctx.push()

		self.inference_fn = do_inference if trt.__version__[0] < '7' else do_inference_v2
		self.trt_logger = trt.Logger(trt.Logger.INFO)
		self.engine = self._load_engine()

		try:
			self.context = self.engine.create_execution_context()
			self.inputs, self.outputs, self.bindings, self.stream, self.device_alloc_memories = \
				allocate_buffers(self.engine)
		except Exception as e:
			raise RuntimeError('fail to allocate CUDA resources') from e
		finally:
			if self.cuda_ctx:
				self.cuda_ctx.pop()

	def __del__(self):
		"""Free CUDA memories."""
		del self.outputs
		del self.inputs
		del self.stream

		del self.context
		del self.engine

		for alloc_mem in self.device_alloc_memories:
			alloc_mem.free()

	def detect_single(self, img, bbox):
		# Preprocess input image
		img_preprocessed, Ms = preprocess_single_image(img, bbox, self.input_shape)

		# Set host input to the image. The do_inference() function
		# will copy the input to the GPU before executing.
		self.inputs[0].host = np.ascontiguousarray(img_preprocessed)
		if self.cuda_ctx:
			self.cuda_ctx.push()
		trt_outputs = self.inference_fn(
			context=self.context,
			bindings=self.bindings,
			inputs=self.inputs,
			outputs=self.outputs,
			stream=self.stream)
		if self.cuda_ctx:
			self.cuda_ctx.pop()

		# Post-process predictions
		hms = trt_outputs[0].reshape(self.reshaping_dims)
		preds = get_preds(hms, Ms, self.input_shape, self.output_shape)

		return preds

	def detect_batch(self, images_preprocessed, Ms):
		# Set host input to the image. The do_inference() function
		# will copy the input to the GPU before executing.
		self.inputs[0].host = np.ascontiguousarray(images_preprocessed)
		if self.cuda_ctx:
			self.cuda_ctx.push()
		trt_outputs = self.inference_fn(
			context=self.context,
			bindings=self.bindings,
			inputs=self.inputs,
			outputs=self.outputs,
			stream=self.stream)
		if self.cuda_ctx:
			self.cuda_ctx.pop()

		# Post-process predictions
		hms = trt_outputs[0].reshape(self.reshaping_dims)
		preds = get_preds(
			hms, Ms, self.input_shape, self.output_shape,
			images_in_batch=len(images_preprocessed)
		)
		return preds


def main():

	# Load model
	model_path = 'models/evopose2d_M_f32_batch32.trt'
	gpu_id = 0
	batch_size = 32
	input_shape = [384, 288, 3]
	output_shape = [192, 144, 17]

	# Load TensorRT EvoPose2D
	ctx = cuda.Device(gpu_id).make_context()
	trt_evo = TrtEVO(
		model_path, input_shape, output_shape,
		batch_size, cuda_ctx=ctx
	)
	print('evo loaded')

	image_filename = 'raw_img.jpg'
	cv2_image = cv2.imread(image_filename)

	bbox = [280.79, 44.73, 218.7, 346.68]

	img_preproc, M = preprocess_single_image(cv2_image, bbox, input_shape)
	normalized_images_list = [img_preproc for i in range(batch_size)]
	Ms_list = [M for i in range(batch_size)]

	normalized_images = tf.stack(normalized_images_list, axis=0)
	Ms = np.concatenate(Ms_list, axis=0)

	print('recognizing...')
	preds = trt_evo.detect_batch(normalized_images, Ms)
	print(preds[-1])


if __name__ == '__main__':
	main()
