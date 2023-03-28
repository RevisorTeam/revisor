import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from datetime import timedelta, datetime
from threading import Thread, Event
from shapely.geometry import Point
import pycuda.driver as cuda
from queue import Queue
from time import sleep
import numpy as np
import threading
import psutil
import shutil
import csv
import cv2
import sys
import gc

from cfg import cfg

from components.person_orientation import get_person_orientation, get_voting_orientation, interpolate_bboxes
from components.visualization import get_caps_overlays, draw_ballot_boxes, draw_voter_bbox
from evopose2d.preprocessing import get_joints_dict, get_joints_confs, prepare_evo_input
from get_videos import get_local_videos, get_boxes_rec_vid
from trackers.tracker_attributes import TrackingVoter
from yolact.utils.functions import MovingAverage

from utils import (
	get_avg_box_coefs, get_voting_day_times, get_voted_ballot_box,
	voter_stopped_near_box, get_yolo_trt_bboxes, del_ballot_boxes,
	append_row_to_csv, get_bbox_params, save_output_json,
	create_folder, read_csv, detect_hardware,
	replace_dict_frame_ids,
)

from utils_video import (
	get_stream_params, frame_drop_needed, get_rec_zone_coords,
	vid_saver, get_cap_polygons, upscale_boxes,
	convert_to_coordinates
)

from components.global_variables import (

	# Global variables:
	processing_boxes_tasks, processing_counting_tasks, taken_gpus,
	processed_joints_queues_info, vote_times, counters, frames_memory,
	evo_batches, ballot_boxes_data, cap_polygons_data,

	# Global locks:
	vote_times_lock, processing_boxes_tasks_lock, processing_counting_tasks_lock,
	frames_memory_lock, counters_lock, taken_gpus_lock, saving_lock,
	processed_joints_queues_info_lock, videos_counted_lock,
	evo_batches_lock, ballot_boxes_lock
)

videos_counted = 0
gc.enable()


def main():

	allowed_modes = ['local', 'api']
	if cfg.data_source not in allowed_modes:
		raise ValueError(
			f'Revisor mode ("{cfg.data_source}") is not allowed'
		)

	if cfg.data_source == 'api':
		raise AssertionError(
			'\nFully automatic "API" mode is not implemented in this version.'
			'\nContact us if you would like to process your data using the Revisor.'
		)

	import torch
	torch.cuda.empty_cache()

	# Create temp, stats and outputs directories if there aren't any
	check_dirs()

	# Create ballot boxes recognition queue, saving votes queue and API queue
	queues, threads, events = initialize_base_queues()
	print('Pipeline | Initialized')

	# Local mode (cfg.data_source='local' and dataset maker modes disabled):
	# 	* Find videos of the target directories listed in target_dirs.csv
	# 	* Detect ballot boxes
	# 	* Count voter turnout
	#
	# Dataset maker mode (cfg.dataset_maker_mode=True):
	# 	* Find videos of the target directories listed in target_dirs.csv
	# 	* Save samples of all persons who intersected ballot box lid zone by hands
	#
	# ReID dataset maker mode (cfg.reid_dataset_maker_mode=True):
	# 	* Find videos of the target directories listed in target_dirs.csv
	# 	* Save samples of all persons who appeared in the frame
	#
	# API mode (not revealed in current version of the Revisor):
	# 	* Get task from API (boxes detection / turnout counting)
	# 	* Do the task
	# 	* Send results to API
	# 	* Get next task...
	process_local_data(queues, threads, events)

	# Clear torch GPU memory
	torch.cuda.empty_cache()

	print('Pipeline | Restarting...')


def process_local_data(queues, threads, events):
	"""
	Count voter turnout of videos stored locally.
	"""

	# Parse target video directories and put required data in "tasks" dict
	tasks = get_local_videos()

	# Detect ballot boxes on each station
	for uik_id, uik in tasks.items():

		# Find video to detect ballot boxes
		video_path = get_boxes_rec_vid(uik['videos'])

		are_boxes, objects_num, objects, objects_images, camera_quality = \
			recognize_ballot_boxes(
				queues, video_path, uik['cam_id'],
				uik['region_number'], uik['station_number'], uik['cam_number'],
				'boxes', uik_id
			)

		tasks[uik_id]['are_boxes'] = are_boxes
		tasks[uik_id]['boxes_number'] = objects_num
		tasks[uik_id]['boxes'] = objects
		tasks[uik_id]['boxes_images'] = objects_images
		tasks[uik_id]['camera_quality'] = camera_quality

	# Start turnout counting
	uiks_threads = {}
	if tasks:

		# Initialize counting threads, models and queues
		print('Pipeline | Initializing counting models...')
		evo_ctx, yolo_ctx, queues, threads, yolo_detector = initialize_counting(
			queues, threads, gpu_id=0
		)
		print('Pipeline | Counting models initialized!')

		for uik_id, task in tasks.items():

			# Make sure we have free counting thread available
			uiks_threads = wait_for_free_slot(uiks_threads)

			# If none boxes are found, skip voting station.
			# In ReID dataset maker mode we do not use ballot boxes.
			if not task['are_boxes'] and not cfg.reid_dataset_maker_mode:
				append_row_to_csv(cfg.processed_videos_csv, [task['target_dir']])
				continue

			# Distribute a unique queue of analyzed
			# pose coordinates data to the task
			distribute_task(uik_id)

			# Create new turnout counting thread
			uiks_threads[uik_id] = count_voters(
				uik_id, task, queues, yolo_ctx, yolo_detector, block=True
			)

		# Join all created threads
		print('Pipeline | Clearing counting models...')
		join_counting(queues, threads)
		del yolo_detector
		clear_gpu_context(yolo_ctx)
		clear_gpu_context(evo_ctx)

	uiks_threads = join_all_counting_threads(uiks_threads)

	print('Pipeline | Joining API...')
	join_pipeline(queues, threads, events, join_counting_pipeline=False)


def recognize_ballot_boxes(
		queues, video_path, cam_id,
		region_number, station_number, cam_number,
		task_type, video_id, gpu_id=0,
		video_info=None, seconds_to_skip=0
):

	are_boxes = None
	objects_num = None
	objects = None
	objects_images = None
	camera_quality = None

	if cfg.reid_dataset_maker_mode:
		return are_boxes, objects_num, objects, objects_images, camera_quality

	print(
		'R {}, UIK {}, CAM {} |\tRecognizing boxes\t| '
		'Recognizing boxes...'.format(
			region_number, station_number, cam_number
		)
	)

	boxes_thread = BoxesDetector(
		queues['find_boxes'], queues['recognized_boxes'],
		queues['api'], gpu_id
	)
	boxes_thread.setDaemon(True)
	boxes_thread.start()

	if video_info is not None:
		seconds_to_skip, _ = get_voting_day_times(video_info)

	with processing_boxes_tasks_lock:
		processing_boxes_tasks.append(cam_id)

	queues['find_boxes'].put((
		task_type, video_id, cam_id,
		video_path, region_number, station_number,
		cam_number, seconds_to_skip
	))

	boxes_thread.join()
	sleep(0.1)

	while True:

		with processing_boxes_tasks_lock:
			processing_boxes_uiks = len(processing_boxes_tasks)

		if queues['find_boxes'].qsize() == 0 and \
			queues['recognized_boxes'].qsize() == 0 and \
			processing_boxes_uiks == 0:
			break

		(are_boxes, objects_num, objects, objects_images, camera_quality) = \
			queues['recognized_boxes'].get()

		sleep(0.05)

	if camera_quality is not None:
		print(
			'R {}, UIK {}, CAM {} |\tRecognizing boxes\t| '
			'Found {} box(-es)! Camera quality {:.2f}%'.format(
				region_number, station_number, cam_number,
				objects_num, camera_quality * 100
			)
		)

	if cfg.data_source == 'api' and camera_quality is None:
		# Post error with boxes recognition to API
		pass

	return are_boxes, objects_num, objects, objects_images, camera_quality


def count_voters(task_id, task, queues, yolo_ctx, yolo_detector, block=False):
	"""
	Start turnout counting thread.

	Args:
		task_id: unique identification number of the task.
		task: dictionary with task data.
		queues: counting queues.
		yolo_ctx: GPU context of the YOLO TRT model.
		yolo_detector: YOLO TRT model object.
		block: whether to block code execution until thread will end processing.
			You should set to True only if cfg.parallel_counting_videos == 1.
			(When the Revisor processes cameras one by one)

	Returns:
		task_thread: created thread's object
	"""

	print(
		'TASK_ID {}, R {}, UIK {}, CAM {} |\tCounting\t| '
		'Preparing for counting...'.format(
			task_id, task['region_number'], task['station_number'],
			task['cam_number']
		)
	)

	task_thread = VotesCounter(
		task_id, task['uik_id'], task, queues, yolo_ctx, yolo_detector
	)
	task_thread.setDaemon(True)
	task_thread.start()

	if block:
		# Wait until turnout counting thread will end processing task's videos
		task_thread.join()
	else:
		# Wait until turnout counting thread initialize
		sleep(10)

	return task_thread


class VotesCounter(threading.Thread):
	"""
	Polling station's camera processing thread
	"""

	def __init__(self, task_id, uik_id, task_info, queues, yolo_ctx, yolo_detector):
		threading.Thread.__init__(self)

		from yolov4_trt.utils.yolo_classes import get_cls_dict

		self.task_id = task_id
		self.uik_id = uik_id
		self.video_id = None
		self.video_path = None
		self.queues = queues
		self.task_info = task_info
		with processed_joints_queues_info_lock:
			self.processed_joints_queues_info = processed_joints_queues_info

		self.ctx = yolo_ctx
		self.yolo_detector = yolo_detector
		self.yolo_classes = get_cls_dict(80)

	def run(self):

		cuda.init()

		global videos_counted

		with evo_batches_lock:
			evo_batches[self.task_id] = {}

		with processing_counting_tasks_lock:
			processing_counting_tasks[self.task_id] = True

		with vote_times_lock:
			vote_times[(
				self.task_id,
				self.uik_id,
				self.task_info['cam_id'],
				self.task_info['region_number'],
				self.task_info['station_number']
			)] = []

		with counters_lock:
			counters[self.task_id] = {
				'votes': 0,
				'backwards_votes': 0,
				'rejected_votes': 0,
				'cap_intersections': 0,
			}

		# Pick the least loaded GPU in the system.
		with taken_gpus_lock:
			gpu_id = min(taken_gpus, key=taken_gpus.get)
			taken_gpus[gpu_id] += 1

		try:

			print(
				'TASK_ID {}, R {}, UIK {}, CAM {} |\tCounting\t| '
				'Thread created'.format(
					self.task_id,
					self.task_info['region_number'],
					self.task_info['station_number'],
					self.task_info['cam_number']
				))

			# Find vote number of current dir - for dataset maker mode only
			if cfg.dataset_maker_mode or cfg.reid_dataset_maker_mode:
				files_list = os.listdir(
					os.path.join(cfg.source_videos_dir, self.task_info['target_dir'])
				)
				if files_list:
					dataset_videos = os.listdir(os.path.join(cfg.dataset_dir, 'videos'))
					current_target_files_name = '{}_{}_{}'.format(
						self.task_info['region_number'], self.task_info['station_number'],
						self.task_info['cam_number']
					)

					existing_votes = []
					for vid_filename in dataset_videos:
						if current_target_files_name in vid_filename:
							existing_vote_num = int(
								(vid_filename.split('_')[-1]).split('.')[0]
							)
							existing_votes.append(existing_vote_num)

					with counters_lock:
						counters[self.task_id]['votes'] = \
							max(existing_votes) + 1 if existing_votes else 0

			if cfg.data_source == 'local':

				region_num, uik_num, camera_num, camera_quality = \
					self.task_info['region_number'], self.task_info['station_number'], \
					self.task_info['cam_number'], self.task_info['camera_quality']
				processed_files_csv_path = os.path.join(
					cfg.processed_videos_dir,
					'{}_{}_{}_processed_videos.csv'.format(
						region_num, uik_num, camera_num
					)
				)

				# Find mean dist and width parameters of ballot boxes
				dist_avg, width_avg = None, None
				if not cfg.reid_dataset_maker_mode:
					dist_avg, width_avg = get_avg_box_coefs(self.task_info['boxes'])

				files_total = len(self.task_info['videos'].keys())
				for files_processed, (file_id, video_info) in enumerate(
					self.task_info['videos'].items()):

					# Add to processed videos csv current video filename.
					# For dataset maker mode only
					if cfg.dataset_maker_mode or cfg.reid_dataset_maker_mode:
						append_row_to_csv(
							processed_files_csv_path,
							[video_info['filename']]
						)

					votes_at_start_vid = counters[self.task_id]['votes']
					rej_votes_at_start_vid = counters[self.task_id]['rejected_votes']
					box_intersections_at_start_vid = counters[self.task_id]['cap_intersections']
					backwards_votes_at_start_vid = counters[self.task_id]['backwards_votes']
					start = datetime.now()

					# Count voter turnout
					try:
						video_stream = cv2.VideoCapture(video_info['path'])
						process_video_stream(
							video_stream=video_stream,
							uik_data=self.task_info,
							file_info=video_info,
							yolo_classes=self.yolo_classes,
							yolo_detector=self.yolo_detector,
							task_id=self.task_id,
							uiks_processed_joints_queues=self.processed_joints_queues_info,
							gpu_id=gpu_id,
							queues=self.queues
						)
					except Exception as e:
						print(
							'Counting local | Skipping video {}. Exception: {}'.format(
								video_info['filename'], e
							)
						)

					votes_per_video = counters[self.task_id]['votes'] - votes_at_start_vid
					rejected_votes_per_video = \
						counters[self.task_id]['rejected_votes'] - rej_votes_at_start_vid
					box_intersections_per_video = \
						counters[self.task_id]['cap_intersections'] - box_intersections_at_start_vid
					backwards_votes_per_video = \
						counters[self.task_id]['backwards_votes'] - backwards_votes_at_start_vid

					processing_time = (datetime.now() - start).total_seconds()

					append_row_to_csv(cfg.stats_filepath, [
						'{}_{}'.format(region_num, uik_num), video_info['filename'],
						votes_per_video, rejected_votes_per_video,
						box_intersections_per_video, counters[self.task_id]['votes'],
						backwards_votes_per_video, camera_quality, dist_avg, width_avg,
						processing_time
					])

					# Add to processed videos csv current video filename
					if not cfg.dataset_maker_mode and not cfg.reid_dataset_maker_mode:
						append_row_to_csv(processed_files_csv_path, [video_info['filename']])

					estimated_time_left = int((files_total - files_processed) * processing_time)
					progress = files_processed / files_total if files_total != 0 else 1.0
					print(
						'TASK_ID {}, R {}, UIK {}, CAM {} |\tCounting\t| '
						'Video processed, Processing time {} sec\n'
						'Progress {:.2f}%, ETA {} sec'.format(
							self.task_id,
							self.task_info['region_number'],
							self.task_info['station_number'],
							self.task_info['cam_number'],
							processing_time,
							progress * 100,
							estimated_time_left
						))

					video_stream.release()

				append_row_to_csv(cfg.processed_videos_csv, [self.task_info['target_dir']])

			elif cfg.data_source == 'api':

				for files_processed, (file_id, video_info) in enumerate(self.task_info['videos'].items()):
					processing_start_time = datetime.now()
					self.video_id, self.video_path = video_info['video_id'], video_info['path']
					video_stream = cv2.VideoCapture(self.video_path)

					# - Find number of seconds to skip from start of the video
					# 		(if video starts earlier than opening hour of the station)
					# - Find second when we need to stop processing
					# 		(if some part of the vid > closing hour of the station)
					seconds_to_skip, stop_time = get_voting_day_times(video_info)

					# Count voter turnout
					process_video_stream(
						video_stream=video_stream,
						uik_data=self.task_info,
						file_info=video_info,
						yolo_classes=self.yolo_classes,
						yolo_detector=self.yolo_detector,
						task_id=self.task_id,
						uiks_processed_joints_queues=self.processed_joints_queues_info,
						gpu_id=gpu_id,
						queues=self.queues,
						seconds_to_skip=seconds_to_skip,
						stop_time=stop_time
					)

					processing_time = (
						datetime.now() - processing_start_time
					).total_seconds()
					video_stream.release()
					print(
						'TASK_ID {}, R {}, UIK {}, CAM {} |\tCounting\t| '
						'Video processed, Processing time {} sec'.format(
							self.task_id,
							self.task_info['region_number'],
							self.task_info['station_number'],
							self.task_info['cam_number'],
							processing_time
						)
					)

		except Exception as e:
			print('Counting | Exception:', e)
			if cfg.data_source == 'api':
				# Post error to API
				pass

		finally:

			with videos_counted_lock:
				videos_counted += 1

			with taken_gpus_lock:
				taken_gpus[gpu_id] -= 1

			if cfg.data_source == 'api':
				# Post to API that current tas has been finished
				pass

			with processed_joints_queues_info_lock:
				if self.task_id in processed_joints_queues_info.keys():
					processed_joints_queues_info.pop(self.task_id)

			# Delete source videos if specified
			if cfg.delete_processed_videos:
				if cfg.data_source == 'local':
					if os.path.exists(self.task_info['videos_dir']):
						shutil.rmtree(self.task_info['videos_dir'])
				elif cfg.data_source == 'api' and self.video_path is not None:
					os.remove(self.video_path)

			with processing_counting_tasks_lock:
				processing_counting_tasks[self.task_id] = False

			gc.collect()

			print(
				'TASK_ID {}, R {}, UIK {}, CAM {} |\tCounting\t| '
				'Task processed! Thread joined'.format(
					self.task_id, self.task_info['region_number'],
					self.task_info['station_number'],
					self.task_info['cam_number']
				)
			)

			sys.exit()


class BoxesDetector(threading.Thread):
	"""
	Ballot boxes recognition thread
	"""

	def __init__(self, camera_to_rec_queue, recognized_boxes_queue, api_queue, gpu_id):
		threading.Thread.__init__(self)
		self.camera_to_rec_queue = camera_to_rec_queue
		self.recognized_boxes_queue = recognized_boxes_queue
		self.api_queue = api_queue
		self.gpu_id = gpu_id

	def run(self):

		from ballot_boxes_finder import find_cam_quality
		import torch

		(
			task_type, video_id, cam_id, video_path,
			region_num, uik_num, camera_num, seconds_to_skip
		) = self.camera_to_rec_queue.get()

		try:
			if cam_id is not None:

				torch.cuda.empty_cache()

				temp_cam_files = os.path.join(
					cfg.temp_dir, '{}_{}'.format(region_num, uik_num)
				)
				boxes, objects_num, objects, objects_images, camera_quality = \
					find_cam_quality(
						camera_num, video_path, temp_cam_files,
						verbose=False, gpu_id=self.gpu_id,
						start_second=seconds_to_skip
					)

				torch.cuda.empty_cache()

				# Remove temp directory if specified
				if cfg.delete_temp_files:
					shutil.rmtree(temp_cam_files)

				self.recognized_boxes_queue.put((
					boxes, objects_num, objects,
					objects_images, camera_quality
				))

		except Exception as e:
			print(
				'Recognizing boxes | '
				'Exception occurred while recognizing ballot boxes:', e
			)

		finally:
			with processing_boxes_tasks_lock:
				processing_boxes_tasks.remove(cam_id)

			self.camera_to_rec_queue.task_done()


class JointsTransformer(threading.Thread):
	"""
	Preprocessing of pose estimation model's inputs.
	"""

	def __init__(self, joints_queue, joints_to_rec_queues):
		threading.Thread.__init__(self)
		self.joints_queue = joints_queue
		self.joints_to_rec_queues = joints_to_rec_queues

		self.poses_max_batches = 2
		self.wait_time = 0.05

	def run(self):

		while True:

			(
				task_id, voter_id, voter_bboxes, first_visible_frame, last_visible_frame,
				resized_width, resized_height, uiks_processed_joints_queues, gpu_id
			) = self.joints_queue.get()

			# Wait until empty slot in preprocessing queue will free up.
			# If pose estimation queue (joints_to_rec_queue) will have
			# a lot of preprocessed data, GPU OOM will be raised.
			while True:
				if voter_id == 'STOP':
					break

				prepared_joints_q_size = self.joints_to_rec_queues[gpu_id].qsize()
				if prepared_joints_q_size < self.poses_max_batches:
					if cfg.log_pose_queue_sizes:
						print('PREP_POSES_Q: {}, POSES_Q: {}'.format(
							self.joints_queue.qsize(), prepared_joints_q_size)
						)
					break
				else:
					if cfg.log_pose_queue_sizes:
						print(
							'Waiting {:.2f} sec for free batch: '
							'evo_prep_q_size {} > max_q_size {}'.format(
								self.wait_time, prepared_joints_q_size,
								self.poses_max_batches
							)
						)
					sleep(self.wait_time)

			try:
				if voter_id is not None and voter_id != 'STOP':

					# Fill missed bbox coordinates with interpolated values
					# (from the nearest bboxes)
					voter_bboxes = interpolate_bboxes(voter_bboxes)

					# Load from global variable frames which will be used for pose estimation
					prepared_frames, min_prep_frame_id, max_prep_frame_id = prepare_voting_vid(
						frames_memory[task_id], first_visible_frame,
						last_visible_frame, out_type='dict'
					)

					if prepared_frames:
						batches_images, batches_Ms, origins, old_frame_ids = \
							prepare_evo_input(
								voter_bboxes, prepared_frames,
								min_prep_frame_id, max_prep_frame_id,
								batch_size=cfg.models.pose.batch_size,
								pose_input_shape=cfg.models.pose.input_shape
							)

						evo_batches[task_id][voter_id] = (
							batches_images, batches_Ms, origins, old_frame_ids
						)

						self.joints_to_rec_queues[gpu_id].put((
							voter_id, uiks_processed_joints_queues,
							task_id, gpu_id
						))

				elif voter_id == 'STOP':
					break

			except Exception as e:
				print('JT Exception!', e)
			finally:
				self.joints_queue.task_done()

		gc.collect()
		sys.exit()


class JointsEstimator(threading.Thread):
	"""
	Pose estimation thread
	"""

	def __init__(
			self, joints_to_rec_queues, joints_to_analyze_queue,
			gpu_id, ctx, api_queue
	):
		threading.Thread.__init__(self)
		self.joints_to_rec_queues = joints_to_rec_queues
		self.joints_to_analyze_queue = joints_to_analyze_queue
		self.gpu_id = gpu_id
		self.ctx = ctx
		self.api_queue = api_queue

	def run(self):

		cuda.init()

		from evopose2d.evo_trt import TrtEVO

		# Initialize EvoPose2D TensorRT engine
		try:
			evo_model = TrtEVO(
				cfg.models.pose.model_path,
				cfg.models.pose.input_shape,
				cfg.models.pose.output_shape,
				batch_size=cfg.models.pose.batch_size,
				cuda_ctx=self.ctx
			)
		except Exception as e:
			evo_model = None

		while True:

			(voter_id, uiks_processed_joints_queues, task_id, gpu_id) = \
				self.joints_to_rec_queues[self.gpu_id].get()

			try:
				if voter_id is not None and voter_id != 'STOP':

					# Receive batches from global variable
					# (that's faster than receiving from queues)
					(batches_images, batches_Ms, origins, old_frame_ids) = \
						evo_batches[task_id][voter_id]

					if cfg.log_pose_processing:
						print(
							'TASK ID {}, GPU ID {} |\tCounting\t| '
							'Joints rec. thread got {} batches from queue. Person #{}'.format(
								task_id, gpu_id, len(batches_Ms), voter_id
							)
						)

					raw_preds = []
					for batch_id, batch_images in enumerate(batches_images):

						pred = evo_model.detect_batch(batch_images, batches_Ms[batch_id])
						raw_preds.append(pred)

					if cfg.log_pose_processing:
						print(
							'TASK ID {}, GPU ID {} |\tCounting\t| '
							'Joints rec. thread processed {} batches of #{} person'.format(
								task_id, gpu_id, len(batches_Ms), voter_id
							)
						)
					self.joints_to_analyze_queue.put((
						voter_id, raw_preds, origins, old_frame_ids,
						uiks_processed_joints_queues, task_id
					))

					del evo_batches[task_id][voter_id]

				elif voter_id == 'STOP':
					break

			except Exception as e:
				print('JE Exception!', e)

				# Send errors to API
				if cfg.data_source == 'api':
					if evo_model is None:
						err_message = 'trt evo loading error'
					else:
						err_message = str(e)

					# Post error to API
					pass

			finally:
				self.joints_to_rec_queues[self.gpu_id].task_done()

		del evo_model
		gc.collect()
		sys.exit()


class JointsAnalyzer(threading.Thread):
	"""
	Joints coordinates analyzing thread - check for hands intersection
	of the ballot box lid
	"""

	def __init__(self, joints_to_analyze_queue, processed_joints_queues):
		threading.Thread.__init__(self)
		self.joints_to_analyze_queue = joints_to_analyze_queue
		self.processed_joints_queues = processed_joints_queues

	def run(self):

		while True:

			(
				voter_id, batch_predictions, origins,
				old_frame_ids, uiks_processed_joints_queues, task_id
			) = self.joints_to_analyze_queue.get()

			try:
				if voter_id is not None:
					intersected_ballot_box, box_intersections_num = [], {}
					voting_start_frame_id, voting_end_frame_id = None, None
					intersected_cap = False
					output_joints, output_confidences = {}, {}
					image_id = 0

					# Repeat origin coordinates to add to them cropped joints
					# coordinates later (for getting source coords)
					origins_repeated = np.repeat(origins, 17, axis=0).reshape(
						(origins.shape[0], 17, 2)
					)

					# Concat evo output to a single array and add cropped coordinates
					batches_predictions_np = np.concatenate(batch_predictions, axis=0)
					transformed_x = batches_predictions_np[:, :, :-2] + origins_repeated[:, :, :1]
					transformed_y = batches_predictions_np[:, :, 1:-1] + origins_repeated[:, :, 1:]
					transformed_xy = np.concatenate(
						(transformed_x, transformed_y), axis=2
					).astype(int)

					confs = batches_predictions_np[:, :, -1]
					for pred_id, pred in enumerate(transformed_xy):

						# Source frame number
						old_fr_id = old_frame_ids[image_id]
						image_id += 1

						# Convert coordinates and confidences to readable format:
						# { 'nose': value, ... }
						output_joints[old_fr_id] = get_joints_dict(pred)
						output_confidences[old_fr_id] = get_joints_confs(confs[pred_id])

						target_joints = [
							output_joints[old_fr_id][t_joint_name] for t_joint_name in cfg.target_joints
						]
						if not cfg.reid_dataset_maker_mode:

							for ballot_box_id, ballot_box_dict in ballot_boxes_data[task_id].items():

								for [t_joint_x, t_joint_y] in target_joints:

									# Detect if a ballot box lid (cap) contains target joint
									joint_point = Point(t_joint_x, t_joint_y)
									target_point_in_ballot_box = \
										cap_polygons_data[task_id][ballot_box_id].contains(joint_point)

									# If a target joint was inside lid zone,
									# save that frame id and intersected ballot box id
									if target_point_in_ballot_box:

										intersected_cap = True

										if ballot_box_id not in intersected_ballot_box:
											intersected_ballot_box.append(ballot_box_id)

										if voting_start_frame_id is None:
											voting_start_frame_id = old_fr_id
										voting_end_frame_id = old_fr_id

								# Update ballot boxes intersections counter
								if intersected_cap:
									if ballot_box_id not in box_intersections_num.keys():
										box_intersections_num[ballot_box_id] = 1
									else:
										box_intersections_num[ballot_box_id] += 1

					# For ReID dataset maker mode all people in frame are collected
					if cfg.reid_dataset_maker_mode:
						intersected_cap = True
						voting_start_frame_id, voting_end_frame_id = \
							min(output_joints.keys()), max(output_joints.keys())

					# Put processed data back to main thread's queue
					self.processed_joints_queues[uiks_processed_joints_queues[task_id]].put((
						voter_id, intersected_cap, output_joints,
						output_confidences, intersected_ballot_box,
						voting_start_frame_id, voting_end_frame_id,
						box_intersections_num
					))

			except Exception as e:
				print('JA Exception!', e)
			finally:
				self.joints_to_analyze_queue.task_done()


class VotesTransformer(threading.Thread):
	"""
	Data collecting thread of person who intersected ballot box lid zone by hands
	"""

	def __init__(self, votes_queue, votes_to_rec_queue, votes_to_analyze_queue):
		threading.Thread.__init__(self)
		self.votes_queue = votes_queue
		self.votes_to_rec_queue = votes_to_rec_queue
		self.votes_to_analyze_queue = votes_to_analyze_queue

	def run(self):

		while True:

			(
				voter_id, vote_info, vid_info,
				file_info, uik_info, task_id
			) = self.votes_queue.get()

			try:
				if voter_id is not None:

					# Get from global variable source frames of tracked person
					if cfg.show_votes_window or cfg.show_rej_votes_window or cfg.save_votes_vid \
							or cfg.save_rejected_votes_vid or cfg.dataset_maker_mode \
							or cfg.reid_dataset_maker_mode:
						prep_output_frames, min_save_frame_id, max_save_frame_id = prepare_voting_vid(
							frames_memory[task_id], vote_info['voting_start_saving_frame_id'],
							vote_info['voting_end_saving_frame_id']
						)
					else:
						prep_output_frames, min_save_frame_id, max_save_frame_id = \
							None, vote_info['voting_start_saving_frame_id'], \
							vote_info['voting_end_saving_frame_id']

					if not cfg.reid_dataset_maker_mode:
						ballot_boxes = ballot_boxes_data[task_id]

					# Replace source frame ids with updated frame ids
					new_joints, new_voting_start, new_voting_end = replace_dict_frame_ids(
						vote_info['joints'],
						min_save_frame_id, max_save_frame_id,
						vote_info['voting_start_frame_id'],
						vote_info['voting_end_frame_id']
					)

					new_joints_confidences, _, __ = replace_dict_frame_ids(
						vote_info['joints_confidences'],
						min_save_frame_id, max_save_frame_id,
						vote_info['voting_start_frame_id'],
						vote_info['voting_end_frame_id']
					)

					new_bboxes, _, __ = replace_dict_frame_ids(
						vote_info['person_bboxes'],
						min_save_frame_id, max_save_frame_id,
						vote_info['voting_start_frame_id'],
						vote_info['voting_end_frame_id']
					)

					new_orientations, _, __ = replace_dict_frame_ids(
						vote_info['orientations'],
						min_save_frame_id, max_save_frame_id,
						vote_info['voting_start_frame_id'],
						vote_info['voting_end_frame_id']
					)

					# Put voted ballot box data to an output dictionary
					output_ballot_box, ballot_box_centroid, output_cap, \
						output_upscaled_cap, cap_centroid = {}, {}, {}, {}, {}
					voted_box_id = vote_info['voted_ballot_box_id']

					if cfg.reid_dataset_maker_mode:
						output_ballot_box, output_cap, output_upscaled_cap, \
							ballot_box_centroid, cap_centroid = \
							{}, {}, {}, {}, {}
					else:
						output_ballot_box[str(voted_box_id)] = {
							'x1': ballot_boxes[voted_box_id]['bbox']['x1'],
							'y1': ballot_boxes[voted_box_id]['bbox']['y1'],
							'x2': ballot_boxes[voted_box_id]['bbox']['x2'],
							'y2': ballot_boxes[voted_box_id]['bbox']['y2']
						}
						output_cap[str(voted_box_id)] = {
							'x1': ballot_boxes[voted_box_id]['caps']['ort_bbox']['x1'],
							'y1': ballot_boxes[voted_box_id]['caps']['ort_bbox']['y1'],
							'x2': ballot_boxes[voted_box_id]['caps']['ort_bbox']['x2'],
							'y2': ballot_boxes[voted_box_id]['caps']['ort_bbox']['y2']
						}
						output_upscaled_cap[str(voted_box_id)] = {
							'x1': ballot_boxes[voted_box_id]['caps']['upscaled_ort_bbox']['x1'],
							'y1': ballot_boxes[voted_box_id]['caps']['upscaled_ort_bbox']['y1'],
							'x2': ballot_boxes[voted_box_id]['caps']['upscaled_ort_bbox']['x2'],
							'y2': ballot_boxes[voted_box_id]['caps']['upscaled_ort_bbox']['y2']
						}
						ballot_box_centroid[str(voted_box_id)] = {
							'x': int(ballot_boxes[voted_box_id]['centroid_k']['x'] * vid_info['resized_width']),
							'y': int(ballot_boxes[voted_box_id]['centroid_k']['y'] * vid_info['resized_height'])
						}
						cap_centroid[str(voted_box_id)] = {
							'x': int(ballot_boxes[voted_box_id]['caps']['centroid_k']['x'] * vid_info['resized_width']),
							'y': int(ballot_boxes[voted_box_id]['caps']['centroid_k']['y'] * vid_info['resized_height'])
						}

					# In API mode ballot box type loaded from API as a str,
					# so we need to convert it to int:
					# 	* 0: 'ballot_box' - regular ballot box
					# 	* 1: 'koib' - electronic ballot box
					if cfg.data_source == 'api':
						if ballot_boxes[voted_box_id]['type'] == 'ballot_box':
							voted_ballot_box_type = 0
						elif ballot_boxes[voted_box_id]['type'] == 'koib':
							voted_ballot_box_type = 1
						else:
							voted_ballot_box_type = 'n/d'

					# In local mode ballot box type loaded from stations json file
					else:
						voted_ballot_box_type = uik_info['box_type'] \
							if not cfg.reid_dataset_maker_mode else None

					voter_dict = {
						'analyzing_src_filename': vid_info['vid_file'],
						'video_fps': vid_info['video_fps'],
						'video_width': vid_info['resized_width'],
						'video_height': vid_info['resized_height'],
						'vote_local_time': vote_info['vote_local_time'],
						'voting_start_frame': new_voting_start,
						'voting_end_frame': new_voting_end,
						'voter_id': voter_id,
						'camera_quality': uik_info['camera_quality']
							if not cfg.reid_dataset_maker_mode else None,
						'camera_id': uik_info['cam_id'],
						'region': uik_info['region_number'],
						'uik': uik_info['station_number'],
						'camera_num': uik_info['cam_number'],
						'ballot_box_bbox': output_ballot_box,
						'voted_ballot_box_type': voted_ballot_box_type,
						'cap_bbox': output_cap,
						'upscaled_cap': output_upscaled_cap,
						'ballot_box_centroid': ballot_box_centroid,
						'cap_centroid': cap_centroid,
						'joint_coords': new_joints,
						'joint_confidences': new_joints_confidences,
						'voter_bboxes': new_bboxes,
						'voter_orientation': new_orientations
					}

					# For dataset maker modes put output dict directly to saving queue
					# (actions will not be recognized)
					if cfg.dataset_maker_mode or cfg.reid_dataset_maker_mode:
						self.votes_to_analyze_queue.put((
							voter_id, 1, 1.0, voter_dict, new_orientations,
							new_voting_start, new_voting_end,
							prep_output_frames, uik_info, vid_info, task_id
						))
					else:
						self.votes_to_rec_queue.put((
							voter_id, voter_dict, new_orientations,
							new_voting_start, new_voting_end,
							prep_output_frames, uik_info, vid_info, task_id
						))

			except Exception as e:
				print('VT Exception!', e)
			finally:
				self.votes_queue.task_done()


class VotesRecognizer(threading.Thread):
	"""
	Action recognition thread - classifies actions on two types:
		* vote
		* other action
	"""

	def __init__(self, votes_to_rec_queue, votes_to_analyze_queue, gpu_id):
		threading.Thread.__init__(self)
		self.votes_to_rec_queue = votes_to_rec_queue
		self.votes_to_analyze_queue = votes_to_analyze_queue
		self.gpu_id = gpu_id

	def run(self):

		import torch
		from action_recognition.votes_recognizer import VotingModel

		action_model = VotingModel(
			checkpoint=cfg.models.action.weights,
			threshold=cfg.models.action.conf_threshold,
			window_size=None
		)

		while True:

			(
				voter_id, voter_dict, new_orientations,
				new_voting_start, new_voting_end,
				prep_output_frames, uik_info, vid_info, task_id
			) = self.votes_to_rec_queue.get()

			try:
				if voter_id is not None and voter_id != 'STOP':

					pred_class, conf = action_model.predict(voter_dict)

					self.votes_to_analyze_queue.put((
						voter_id, pred_class, conf, voter_dict,
						new_orientations, new_voting_start, new_voting_end,
						prep_output_frames, uik_info, vid_info, task_id
					))

				elif voter_id == 'STOP':
					break

			except Exception as e:
				print('VR Exception!', e)
			finally:
				self.votes_to_rec_queue.task_done()

		del action_model
		torch.cuda.empty_cache()
		gc.collect()
		sys.exit()


class VotesAnalyzer(threading.Thread):
	"""
	Saving output jsons / videos thread
	"""

	def __init__(self, votes_to_analyze_queue):
		threading.Thread.__init__(self)
		self.votes_to_analyze_queue = votes_to_analyze_queue

	def run(self):

		while True:

			(
				voter_id, pred_class, conf, voter_dict,
				new_orientations, new_voting_start, new_voting_end,
				prep_output_frames, uik_info, vid_info, task_id
			) = self.votes_to_analyze_queue.get()

			try:
				votes_num, rejected_votes_num = 0, 0
				voter_orientation_name = ''
				if voter_id is not None:
					if pred_class == 1:
						vote = True

						# Find person orientation relative to the camera
						voter_orientation = get_voting_orientation(
							new_orientations, new_voting_start, new_voting_end
						)

						voter_orientation_name = voter_orientation['voting_orientation']
						if voter_orientation['voting_orientation'] == 'back':
							with counters_lock:
								counters[task_id]['backwards_votes'] += 1

						with counters_lock:
							counters[task_id]['votes'] += 1
							votes_num = counters[task_id]['votes']
					else:
						vote = False
						with counters_lock:
							counters[task_id]['rejected_votes'] += 1
							rejected_votes_num = counters[task_id]['rejected_votes']

					voter_dict['vote'] = 1 if vote else 0
					voter_dict['vote_conf'] = str(conf)
					vote_time_str = voter_dict['vote_local_time'].strftime("%Y-%m-%d %H:%M:%S.%f")

					if cfg.log_votes:
						print(
							'TASK ID {}, UIK ID {}\t|\tCounting\t| '
							'Person #{}. Vote: {}, conf: {:.2f}%, local time: {}'.format(
								task_id, uik_info['uik_id'], voter_id, vote,
								float(conf) * 100, vote_time_str
							)
						)

					# Put vote docs into the global vote_times variable (in API mode only):
					# 	* vote local datetime
					# 	* vote confidence (0 ... 100)
					# 	* person orientation name
					if (cfg.data_source == 'api' or cfg.save_votes_times) and vote:
						with vote_times_lock:
							vote_times[(
								task_id, uik_info['uik_id'], voter_dict['camera_id'],
								uik_info['region_number'], uik_info['station_number']
							)].append({
								'vote_datetime': vote_time_str,
								'vote_conf': round(float(conf) * 100, 4),
								'vote_orientation_name': voter_orientation_name
							})

					voter_dict['vote_local_time'] = vote_time_str

					if (cfg.show_votes_window or cfg.show_rej_votes_window
						or cfg.save_votes_vid or cfg.save_rejected_votes_vid) \
						and not cfg.dataset_maker_mode and not cfg.reid_dataset_maker_mode:
						conf_str = '{:.0f}'.format(conf * 100).replace('.', ',')
						output_video_filename = '{}_{}_{}_{}{}_{}.mp4'.format(
							uik_info['region_number'],
							uik_info['station_number'],
							uik_info['cam_number'],
							'vote' if vote else 'rejected',
							votes_num if vote else rejected_votes_num,
							conf_str
						)
						output_video_path = os.path.join(
							cfg.votes_vid_dir if vote else
								cfg.rejected_votes_vid_dir, output_video_filename
						)

						window_header = 'Accepted vote action {}. Person {}. Confidence: {:.2f}%'.format(
							votes_num, voter_id, conf * 100
						) if vote \
							else 'Rejected vote action. Person {}. Confidence: {:.2f}%'.format(
							voter_id, conf * 100
						)

						try:
							with saving_lock:
								vid_saver(
									prep_output_frames, voter_dict,
									draw_recs=cfg.draw_skeletons,
									show_window=cfg.show_votes_window if vote else cfg.show_rej_votes_window,
									save_vid=cfg.save_votes_vid if vote else cfg.save_rejected_votes_vid,
									output_video_path=output_video_path,
									window_name=window_header,
									fps=vid_info['video_fps'],
									width=vid_info['resized_width'],
									height=vid_info['resized_height']
								)
						except Exception as e:
							print('Saving video exception:', e)

					if (cfg.save_votes_json or cfg.save_rejected_json) \
						and not cfg.dataset_maker_mode and not cfg.reid_dataset_maker_mode:
						output_json_filename = '{}_{}_{}_{}{}.json'.format(
							uik_info['region_number'],
							uik_info['station_number'],
							uik_info['cam_number'],
							'vote' if vote else 'rejected',
							votes_num if vote else rejected_votes_num
						)

						# Convert coordinates object from numpy to list type (for saving it to json)
						prepared_coords = {}
						for frame_id, joints in voter_dict['joint_coords'].items():
							if frame_id not in prepared_coords.keys():
								prepared_coords[frame_id] = {}
							for joint_name, joint_coords_np in joints.items():
								prepared_coords[frame_id][joint_name] = joint_coords_np.tolist()
						voter_dict['joint_coords'] = prepared_coords

						output_json_path = os.path.join(cfg.votes_json_dir, output_json_filename)
						with saving_lock:
							save_output_json(
								output_json_path, voter_dict,
								save_json=cfg.save_votes_json if vote else cfg.save_rejected_json
							)

					# In the dataset maker mode save source and
					# labelled videos + corresponding json.
					if cfg.dataset_maker_mode or cfg.reid_dataset_maker_mode:

						# Convert coordinates object from numpy to list type
						prepared_coords = {}
						for frame_id, joints in voter_dict['joint_coords'].items():
							if frame_id not in prepared_coords.keys():
								prepared_coords[frame_id] = {}
							for joint_name, joint_coords_np in joints.items():
								prepared_coords[frame_id][joint_name] = joint_coords_np.tolist()
						voter_dict['joint_coords'] = prepared_coords

						output_video_filename = '{}_{}_{}_{}.mp4'.format(
							uik_info['region_number'], uik_info['station_number'],
							uik_info['cam_number'], votes_num
						)
						output_video_path = os.path.join(cfg.dataset_dir, 'videos', output_video_filename)
						output_labelled_video_path = os.path.join(
							cfg.dataset_dir, 'labelled_videos', output_video_filename
						)
						output_json_filename = '{}_{}_{}_{}.json'.format(
							uik_info['region_number'], uik_info['station_number'],
							uik_info['cam_number'], votes_num
						)
						output_json_path = os.path.join(cfg.dataset_dir, 'json', output_json_filename)

						try:
							with saving_lock:
								vid_saver(
									prep_output_frames,
									voter_dict,
									draw_recs=False,
									save_vid=True,
									output_video_path=output_video_path,
									fps=vid_info['video_fps'],
									width=vid_info['resized_width'],
									height=vid_info['resized_height']
								)
								sleep(0.2)
								if cfg.save_labelled_videos:
									vid_saver(
										prep_output_frames,
										voter_dict,
										draw_recs=True,
										save_vid=True,
										output_video_path=output_labelled_video_path,
										fps=vid_info['video_fps'],
										width=vid_info['resized_width'],
										height=vid_info['resized_height']
									)
									sleep(0.2)
								save_output_json(output_json_path, voter_dict)
						except Exception as e:
							print('Saving video exception:', e)

			except Exception as e:
				print('VA Exception!', e)
			finally:
				self.votes_to_analyze_queue.task_done()


class API(threading.Thread):
	"""
	Thread for sending POST requests to API
	"""

	def __init__(self, api_queue, api_token):
		threading.Thread.__init__(self)
		self.api_queue = api_queue
		self.api_token = api_token

	def run(self):

		while True:
			(query_type, message) = self.api_queue.get()
			try:
				# Process received API post request
				# ...
				pass
			except Exception as e:
				print('API thread | Exception!', e)
			finally:
				self.api_queue.task_done()


class SendVotes(Thread):
	"""
	Thread for monitoring votes in the global variable 'vote_times'.
	If a new vote appears in the vote_times, it will be:
		* in local mode - saved into the votes.csv
		* in API mode - sent to the database via the API
	"""

	def __init__(self, event, wait_time, api_queue):
		Thread.__init__(self)
		self.stop_sending = event
		self.wait_time = wait_time
		self.api_queue = api_queue
		self.query_type = 'api_method_name'

	def run(self):

		while not self.stop_sending.wait(self.wait_time):

			if not cfg.data_source == 'api' and not cfg.save_votes_times:
				continue

			self.send_votes()

		# Before stopping thread make sure all votes are sent
		self.send_votes()

	def send_votes(self):
		with vote_times_lock:
			votes_to_del = []
			for (task_id, uik_id, cam_id, reg_num, uik_num), votes in vote_times.items():
				if not votes:
					continue

				if cfg.send_data:
					# Send recognized votes to API
					message = {'request_payload'}
					self.api_queue.put((self.query_type, message))

				# Save votes to csv
				if cfg.save_votes_times:
					for vote in votes:
						append_row_to_csv(
							'{}/votes.csv'.format(cfg.stats_dir),
							[reg_num, uik_num, vote['vote_datetime'], vote['vote_conf']]
						)

				votes_to_del.append((task_id, uik_id, cam_id, reg_num, uik_num))

			# Empty sent votes
			for vote_to_del in votes_to_del:
				vote_times[vote_to_del] = []


def process_video_stream(
		video_stream: cv2.VideoCapture,
		uik_data: dict,
		file_info: dict,
		yolo_classes: dict,
		yolo_detector,
		task_id: int,
		uiks_processed_joints_queues: dict,
		gpu_id: int,
		queues: dict,
		seconds_to_skip: int = 0,
		stop_time: int = 0,
) -> None:

	joints_queue = queues['joints_transform']
	processed_joints_queue = queues['processed_joints'][uiks_processed_joints_queues[task_id]]
	votes_queue = queues['votes_transform']

	# Initialize person tracker
	from trackers.tracker_sort import Sort
	tracker = Sort(max_age=20, min_hits=1)

	raw_frame_id = 0
	series_frame_id = 0

	tracking_voters = {}
	with frames_memory_lock:
		frames_memory[task_id] = {}

	# Average FPS counter
	frame_times = MovingAverage(100) if cfg.show_recognitions else []

	got_frame, frame = video_stream.read()
	# Read next frame of the video (+ corrupted starting frames fix)
	if not got_frame:
		while not got_frame:
			got_frame, source_frame = video_stream.read()
	process = True if got_frame else False
	raw_frame_id += 1
	series_frame_id += 1

	vid_file = file_info['filename']
	ballot_boxes = uik_data['boxes']

	# Find parameters of source video:
	# 	* FPS
	# 	* Width and height
	# 	* Resized width and height - total amount of pixels
	# 		will be equal to cfg.target_pixels
	# 	* Aspect ratio (fraction of height to width)
	# 	* total number of frames
	raw_video_fps, source_height, source_width, resized_height, resized_width, aspect_ratio = \
		get_stream_params(video_stream, frame, cfg.target_pixels)
	number_of_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
	video_length_secs = int(number_of_frames / raw_video_fps)

	resize_needed = True if source_height * source_width > cfg.target_pixels else False
	if not resize_needed:
		resized_height, resized_width = source_height, source_width

	# Find number of frame we need to process.
	# Only cfg.target_fps frames will be processed - other dropped.
	frames_drop_needed, target_frames, target_fps_ratio, video_fps = \
		frame_drop_needed(raw_video_fps, cfg.target_fps)

	print(
		"TASK ID {}, GPU ID {} |\tVideo docs\t| "
		"Converted video: ({}x{}), FPS: {}.\n"
		"Duration: {} sec. Number of frames to process: "
		"{}\nResize needed: {}, Frames drop needed: {}".format(
			task_id, gpu_id, resized_width, resized_height, video_fps,
			video_length_secs, video_length_secs * video_fps,
			resize_needed, frames_drop_needed
		))

	# Start processing from seconds_to_skip second (from the opening
	# of the polling station at 8:00).
	# If it's not specified - start from beginning.
	frames_to_skip = 0
	if seconds_to_skip != 0:
		frames_to_skip = int(seconds_to_skip * raw_video_fps)
		print('Skipping frames to {}. Second: {}'.format(frames_to_skip, seconds_to_skip))

		video_stream.set(1, frames_to_skip)
		got_frame, frame = video_stream.read()
		if not got_frame:
			while not got_frame:
				got_frame, source_frame = video_stream.read()
		raw_frame_id += 1
		series_frame_id += 1

	# Early stopping frame. If video ends after the closing
	# of the polling station at 20:00.
	stop_frame = stop_time * raw_video_fps

	# Total number of frames to process
	number_of_frames_to_process = number_of_frames - stop_frame - frames_to_skip

	# After this frame we will join pose estimation queue (on each next frame)
	joining_frame = number_of_frames_to_process - (2 * video_fps)

	# After vanish_frames have been elapsed, we assume that person went out of the frame
	vanish_frames = int(cfg.vanish_seconds * video_fps)

	# Number of last N frames stored in RAM
	frames_memory_threshold = int(cfg.frames_memory_seconds * video_fps)

	# Min number of frames when hand of a person must be inside the ballot box lid.
	# If the hand is inside the zone more than min_voting_frames,
	# tracker will be sent to action recognition.
	min_voting_frames = int(cfg.min_voting_time * video_fps)

	# Frames shift of an action. Processed video will be increased
	# by +- vote_shift_frames frames.
	vote_shift_frames = int(cfg.vote_shift_time * video_fps)

	if cfg.save_processing_frames:
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		output_video_path = os.path.join(
			cfg.results_dir, '{}_{}_{}'.format(
				uik_data['region_number'], uik_data['station_number'], vid_file
			)
		)
		vid_writer = cv2.VideoWriter(
			output_video_path, fourcc, video_fps, (resized_width, resized_height)
		)

	# Transform all coefficients to coordinates (for API mode only)
	# TODO The same transforms for local mode?
	if cfg.data_source == 'api':
		ballot_boxes = convert_to_coordinates(ballot_boxes, resized_width, resized_height)

	cap_polygons = None
	cap_centroids_coords = None
	accepted_box_distances = None
	ballot_boxes_overlays = None
	if not cfg.reid_dataset_maker_mode:

		# Upscale ballot box lid zone
		ballot_boxes = upscale_boxes(ballot_boxes, resized_width, resized_height)

		# Delete ballot boxes if there are 'koib' at the station (if specified)
		if cfg.delete_ballot_boxes:
			ballot_boxes = del_ballot_boxes(ballot_boxes)

		# Get shapely objects with ballot boxes lids (caps)
		cap_polygons, cap_centroids_coords = get_cap_polygons(
			ballot_boxes, resized_width, resized_height
		)

		with ballot_boxes_lock:
			ballot_boxes_data[task_id] = ballot_boxes
			cap_polygons_data[task_id] = cap_polygons

		# Compute threshold of accepting person as a stopped near the ballot box
		accepted_box_distances, ballot_boxes_overlays = get_caps_overlays(
			ballot_boxes, cap_polygons
		)

	print(
		'TASK ID {}, GPU ID {} |\tCounting\t| '
		'Preparation ended, starting processing...'.format(task_id, gpu_id)
	)

	# Main cycle of processing video
	background = np.zeros((resized_height, resized_width, 3), np.uint8)
	process_visible_trackers = False
	black_box_ready = False
	frame_id = 0
	while process:

		if series_frame_id in target_frames:

			# Measure frame processing time if cfg.show_recognitions is enabled.
			# In order to put text to the cv2 frame.
			if cfg.show_recognitions:
				processing_start = datetime.now()

			if resize_needed:
				frame = cv2.resize(
					frame, (resized_width, resized_height),
					interpolation=cv2.INTER_AREA
				)

			# Save current source frame to global variable (to RAM)
			with frames_memory_lock:
				frames_memory[task_id][frame_id] = frame.copy()

			# Fill black everything that is far from the ballot box
			if not cfg.reid_dataset_maker_mode:
				if not black_box_ready:
					contour_coords = get_rec_zone_coords(ballot_boxes, resized_width, resized_height)
					contours = [np.array(contour_coords)]
					cv2.fillPoly(background, contours, [255, 255, 255])
					black_box_ready = True
				frame = cv2.bitwise_and(frame, background)

			# Detect people
			pred_bboxes, pred_confidences, pred_classes_ids = yolo_detector.detect(frame, conf_th=0.3)

			# Get data with analyzed person hands movements - whether hands intersected
			# ballot box lid zone
			if processed_joints_queue.qsize() != 0:
				(
					processed_voter_id, intersected_cap, voter_joints,
					joints_confidences, intersected_ballot_box,
					voting_start_frame_id, voting_end_frame_id,
					box_intersections_num
				) = processed_joints_queue.get()

				tv = tracking_voters.get(processed_voter_id, None)
				if tv is not None:
					# Assign data to tracker object
					tv.intersected_cap, tv.joints, tv.joints_confidences, tv.intersected_ballot_box, \
					tv.voting_start_frame_id, tv.voting_end_frame_id, tv.box_intersections_num = \
						intersected_cap, voter_joints, joints_confidences, intersected_ballot_box, \
						voting_start_frame_id, voting_end_frame_id, box_intersections_num

					tv.joints_processed = True
					tv.recognizing_joints = False
					tracking_voters[processed_voter_id] = tv
				else:
					print(
						'TASK ID {}, GPU ID {} |\tCounting\t| '
						'Cant get tracking object. Person #{}'.format(
							task_id, gpu_id, processed_voter_id)
					)
				processed_joints_queue.task_done()

			# Iterate over all visible tracking objects to:
			# 	* find trackers that have left the frame and set visible=False for them
			# 	* detect ballot boxes lid intersection
			# 	* put trackers which intersected lid zone in action recognition queue
			voters_to_del = []
			for voter_id in tracking_voters.keys():
				tv = tracking_voters.get(voter_id, None)

				# Count number of frames have passed after last person detection
				frames_past_after_last_voter_det = frame_id - tv.last_visible_frame

				if tv.visible and (frames_past_after_last_voter_det > vanish_frames):
					tv.visible = False

				if process_visible_trackers and not tv.joints_processed and not tv.recognizing_joints and tv.bboxes:

					start_pose_frame_id, end_pose_frame_id = min(tv.bboxes.keys()), max(tv.bboxes.keys())
					appeared_frames_num = end_pose_frame_id - start_pose_frame_id
					if appeared_frames_num > cfg.max_near_box_frames:
						end_pose_frame_id = start_pose_frame_id + cfg.max_near_box_frames

					if appeared_frames_num > 40:  # 5 sec
						joints_queue.put((
							task_id, voter_id, tv.bboxes,
							start_pose_frame_id, end_pose_frame_id,
							resized_width, resized_height,
							uiks_processed_joints_queues, gpu_id
						))
						tv.recognizing_joints = True

				# - Estimate pose for following trackers:
				# 		1. Tracker stopped near the ballot box
				# 		2. Tracker disappeared from the frame
				# 		3. Tracker is not in the pose estimation queue
				# 		4. Pose of that tracker has not been estimated
				# - Check hands intersection of the ballot box lid zone
				#
				# In ReID dataset maker mode: estimate pose of all trackers in the frame
				if not tv.visible:

					if cfg.reid_dataset_maker_mode and not tv.joints_processed \
						and not tv.recognizing_joints and tv.bboxes:

						start_pose_frame_id = min(tv.bboxes.keys())
						end_pose_frame_id = max(tv.bboxes.keys())
						appeared_frames_num = end_pose_frame_id - start_pose_frame_id
						if appeared_frames_num > cfg.max_near_box_frames:
							end_pose_frame_id = start_pose_frame_id + cfg.max_near_box_frames

						if appeared_frames_num > 40:		# 5 sec
							joints_queue.put((
								task_id, voter_id, tv.bboxes,
								start_pose_frame_id, end_pose_frame_id,
								resized_width, resized_height,
								uiks_processed_joints_queues, gpu_id
							))
							tv.recognizing_joints = True

					if tv.stopped_near_box and not tv.joints_processed \
						and not tv.recognizing_joints:

						start_pose_frame_id = min(tv.near_box_frames)
						end_pose_frame_id = max(tv.near_box_frames)

						# Total number of tracker's frames to estimate pose
						near_box_frames_num = end_pose_frame_id - start_pose_frame_id

						# Pick first cfg.max_near_box_frames frames to estimate pose
						if near_box_frames_num > cfg.max_near_box_frames:
							end_pose_frame_id = start_pose_frame_id + cfg.max_near_box_frames

						# Tracker must be more than cfg.min_near_box_frames near ballot box
						if near_box_frames_num > cfg.min_near_box_frames:

							joints_queue.put((
								task_id, voter_id, tv.bboxes,
								start_pose_frame_id, end_pose_frame_id,
								resized_width, resized_height,
								uiks_processed_joints_queues, gpu_id
							))
							tv.recognizing_joints = True

							if cfg.log_pose_processing:
								print(
									'TASK ID {}, GPU ID {} |\tCounting\t| '
									'Put coords of person #{} to joints recognition queue'.format(
										task_id, gpu_id, voter_id
									)
								)

				# 1. Prepare tracker's data:
				# 	* if it's not visible (has gone from the frame)
				# 	* if hands were in the ballot box lid zone more than min_voting_frames frames
				# 2. Put tracker to action recognition queue
				if not tv.visible and tv.intersected_cap \
					and frames_past_after_last_voter_det > vanish_frames * 1.5:

					# Find total number of frames when tracker's hand were inside the lid zone
					hands_intersections_frames = max(
						[intersections_num for obj_id, intersections_num
							in tv.box_intersections_num.items()]
					) if tv.box_intersections_num else 0

					# In ReID dataset maker mode put all trackers to the queue
					if hands_intersections_frames > min_voting_frames \
						or cfg.reid_dataset_maker_mode:

						with counters_lock:
							counters[task_id]['cap_intersections'] += 1
						shift_minus = vote_shift_frames \
							if tv.voting_start_frame_id > vote_shift_frames \
							else tv.voting_start_frame_id
						shift_plus = vote_shift_frames

						# Find tracker's starting and ending frames of the possible voting action
						min_vote_saving_frame_id = tv.voting_start_frame_id - shift_minus
						max_vote_saving_frame_id = tv.voting_end_frame_id + shift_plus
						voting_start_frame_id = tv.voting_start_frame_id
						voting_end_frame_id = tv.voting_end_frame_id

						# Take first N frames if there are too many
						if max_vote_saving_frame_id - min_vote_saving_frame_id > cfg.max_voting_frames:

							max_vote_saving_frame_id = min_vote_saving_frame_id + cfg.max_voting_frames - 1

							if voting_end_frame_id > max_vote_saving_frame_id:
								voting_end_frame_id = max_vote_saving_frame_id

						# Compute person orientation relative to the camera
						# based on pose coordinates (angle, orientation_name).
						output_joints_dict = {}
						output_orientation_dict = {}
						output_joints_confs_dict = {}
						for joint_frame_id, joints in tv.joints.items():
							if min_vote_saving_frame_id <= joint_frame_id <= max_vote_saving_frame_id:
								output_joints_dict[joint_frame_id] = joints
								output_joints_confs_dict[joint_frame_id] = \
									tv.joints_confidences[joint_frame_id]

								if joints:
									output_orientation_dict[joint_frame_id] = get_person_orientation(
										joints, resized_width, resized_height
									) if 'left_shoulder' in joints and 'right_shoulder' in joints else {}
								else:
									output_orientation_dict[joint_frame_id] = {}

						# Get bbox coordinates of target frames
						output_bboxes_dict = {}
						for bbox_frame_id, bbox in tv.bboxes.items():
							if min_vote_saving_frame_id <= bbox_frame_id <= max_vote_saving_frame_id:
								output_bboxes_dict[bbox_frame_id] = bbox

						if not cfg.reid_dataset_maker_mode:
							# Find voted ballot box. Polling station may have 1+ ballot boxes,
							# so we need to find voted one
							tv.voted_ballot_box = get_voted_ballot_box(
								output_joints_dict, cap_centroids_coords,
								voting_start_frame_id, voting_end_frame_id,
								tv.box_intersections_num
							) if len(ballot_boxes.keys()) > 1 else tv.intersected_ballot_box[0]

						vote_info = {
							'voting_start_saving_frame_id': min_vote_saving_frame_id,
							'voting_end_saving_frame_id': max_vote_saving_frame_id,
							'voting_start_frame_id': voting_start_frame_id,
							'voting_end_frame_id': voting_end_frame_id,
							'shift_minus': shift_minus,
							'shift_plus': shift_plus,
							'joints': output_joints_dict,
							'joints_confidences': output_joints_confs_dict,
							'orientations': output_orientation_dict,
							'person_bboxes': output_bboxes_dict,
							'voted_ballot_box_id': tv.voted_ballot_box,
							'vote_local_time': file_info['start_time'] + timedelta(
								seconds=seconds_to_skip) + timedelta(
								seconds=int(voting_end_frame_id / video_fps))
						}

						vid_info = {
							'vid_file': vid_file,
							'video_fps': video_fps,
							'resized_width': resized_width,
							'resized_height': resized_height,
						}

						votes_queue.put((
							voter_id, vote_info, vid_info,
							file_info, uik_data, task_id
						))

						# Add tracker ID to the removing list
						voters_to_del.append(voter_id)

				# Add to removing list trackers that disappeared and didn't vote
				elif not tv.visible and not tv.recognizing_joints \
					and frames_past_after_last_voter_det > vanish_frames * 4:

					if tv.joints_processed and cfg.log_pose_processing:
						print(
							'TASK ID {}, GPU ID {} |\tCounting\t| '
							'Person #{} did not intersect ballot box cap area'.format(
								task_id, gpu_id, tv.voter_id)
						)

					voters_to_del.append(voter_id)

				tracking_voters[voter_id] = tv

			# Delete processed trackers to free up RAM
			for v_id in voters_to_del:
				tracking_voters.pop(v_id)

			# Find person bboxes in the current frame
			detections = get_yolo_trt_bboxes(
				pred_bboxes, pred_confidences, pred_classes_ids, yolo_classes
			)

			# Update tracker with detected bboxes
			voters = tracker.update(detections).tolist()

			# Iterate over each updated tracker object
			for voter in voters:

				voter_x1, voter_y1, voter_x2, voter_y2, voter_id = \
					int(voter[0]), int(voter[1]), int(voter[2]), int(voter[3]), int(voter[4])

				tv = tracking_voters.get(voter_id, None)

				# If unseen tracker is found, create a new tracking object
				if tv is None:
					tv = TrackingVoter(tracking_voters, [voter_x1, voter_y1, voter_x2, voter_y2])
					tv.voter_id = voter_id
					tv.first_visible_frame = frame_id
					tv.visible = True

				# If the tracker already exists, update its coordinates
				else:
					tv.bbox = {'x1': voter_x1, 'y1': voter_y1, 'x2': voter_x2, 'y2': voter_y2}
					tv.centroid, tv.bbox_width, tv.bbox_height = get_bbox_params(
						tv.bbox['x1'], tv.bbox['y1'],
						tv.bbox['x2'], tv.bbox['y2'],
						out_type='int'
					)
					tv.centroids.append(tv.centroid)
					tv.bboxes[frame_id] = tv.bbox

				# Set current frame_id as a last frame when the tracker is seen
				tv.last_visible_frame = frame_id

				# 1. Detect stopping near the ballot boxes
				# 2. Count total number of frames when tracker has stopped near ballot box
				if not cfg.reid_dataset_maker_mode:
					if frame_id % cfg.centroids_analyzing_frame == 0:

						stopped_near_box, near_box = voter_stopped_near_box(
							ballot_boxes, cap_centroids_coords,
							accepted_box_distances,
							tv.centroids, tv.centroid, tv.bbox_width,
							frame if cfg.show_recognitions else None
						)

						if stopped_near_box and not tv.stopped_near_box:
							tv.stopped_near_box = True

						if near_box:
							tv.near_box_frames.append(frame_id)

				# Draw tracker bbox
				if cfg.show_recognitions:
					frame = draw_voter_bbox(
						frame, tv.stopped_near_box,
						tv.bbox, tv.centroid, voter_id
					)

				# Save updated tracker in the trackers dictionary
				tracking_voters[voter_id] = tv

			# Draw ballot boxes, boxes lids and stopping zone
			if cfg.show_recognitions and not cfg.reid_dataset_maker_mode:
				frame = draw_ballot_boxes(
					frame, ballot_boxes, ballot_boxes_overlays, cap_centroids_coords,
					accepted_box_distances, resized_width, resized_height
				)

			# Clear old frames from the frames storage object.
			#
			# If you would like to store all video frames in it,
			# make sure you have enough RAM. Otherwise, you will get RAM OOM.
			with frames_memory_lock:
				min_frame_in_memory = min(list(frames_memory[task_id].keys()))
				frames_in_memory = frame_id - min_frame_in_memory
				if frames_in_memory > frames_memory_threshold:
					del frames_memory[task_id][min_frame_in_memory]

			if cfg.show_recognitions:
				# If cfg.show_recognitions is enabled compute and draw following:
				# 	1. FPS - Average number of processing frames per second.
				# 	2. Playback FPS - The Revisor drops some source frames
				# 		in order to speed up processing, so the playback FPS
				# 		may be higher than FPS if source video has > 8 fps.
				# 	3. Total number of processed frames.
				frame_times.add((datetime.now() - processing_start).total_seconds())
				fps = 1 / frame_times.get_avg()
				playback_fps = fps * target_fps_ratio
				seconds_processed = int(frame_id / cfg.target_fps)

				text_label = [
					"Seconds processed: {}".format(seconds_processed),
					'Playback FPS: {:.2f}'.format(playback_fps),
					'FPS: {:.2f}'.format(fps),
					'Lid intersections: {}'.format(counters[task_id]['cap_intersections']),
					'Rejected actions: {}'.format(counters[task_id]['rejected_votes']),
					'Votes: {}'.format(counters[task_id]['votes'])
				]
				for i, k in enumerate(text_label[::-1]):
					text = "{}".format(k)
					cv2.putText(
						frame, text, (15, resized_height - ((i * 20) + 20)),
						cv2.FONT_HERSHEY_SIMPLEX, 0.4, (212, 255, 255), 1
					)

			if cfg.save_processing_frames:
				vid_writer.write(frame)

			if cfg.show_recognitions:
				cv2.imshow("frame", frame)

			frame_id += 1

		# If there are too much frames in the joints queue,
		# slow down main thread a little.
		joints_queue_size = joints_queue.qsize()
		if joints_queue_size >= 5:
			sleep_time = 0.08
			if joints_queue_size >= 10:
				sleep_time = 0.12
				if joints_queue_size >= 15:
					sleep_time = 0.4
					if joints_queue_size >= 20:
						sleep_time = 1
			sleep(sleep_time)

		# During ending the video start joining queues
		if raw_frame_id > joining_frame:
			if cfg.reid_dataset_maker_mode:
				process_visible_trackers = True
			queues['joints_transform'].join()
			for q_gpu_id, q in queues['joints_to_rec_queues'].items():
				q.join()
			queues['joints_to_analyze'].join()

		# Read the next frame
		got_frame, frame = video_stream.read()
		process = True if got_frame else False

		raw_frame_id += 1
		series_frame_id += 1

		if series_frame_id == raw_video_fps:
			series_frame_id = 0

		# Stop processing if the polling station is closes
		if stop_frame != 0 and raw_frame_id >= stop_frame:
			process = False

		if cfg.show_recognitions and cv2.waitKey(1) & 0xFF == ord('q'):
			process = False

	cv2.destroyAllWindows()
	if cfg.save_processing_frames:
		vid_writer.release()

	# Join all queues
	queues['joints_to_analyze'].join()
	queues['votes_transform'].join()
	queues['votes_to_rec'].join()
	queues['votes_to_analyze'].join()

	# Delete global variables
	with frames_memory_lock:
		del frames_memory[task_id]
	del tracking_voters

	if not cfg.reid_dataset_maker_mode:
		with ballot_boxes_lock:
			del ballot_boxes_data[task_id]
			del cap_polygons_data[task_id]

	gc.collect()


def prepare_voting_vid(
		frames_memory_dict,
		min_frame_id, max_frame_id,
		out_type='list'
):
	"""
	Return source frames of tracked person, min and max source frame ids
	"""

	prepared_frames = [] if out_type == 'list' else {}
	frame_ids = []
	with frames_memory_lock:

		for memory_frame_id in range(min_frame_id, max_frame_id + 1):
			if memory_frame_id in frames_memory_dict.keys():
				if out_type == 'list':
					prepared_frames.append(frames_memory_dict[memory_frame_id])
				else:
					prepared_frames[memory_frame_id] = frames_memory_dict[memory_frame_id]

				frame_ids.append(memory_frame_id)

	min_prep_frame_id, max_prep_frame_id = None, None
	if frame_ids:
		min_prep_frame_id, max_prep_frame_id = min(list(frame_ids)), max(list(frame_ids))

	return prepared_frames, min_prep_frame_id, max_prep_frame_id


def initialize_counting(queues, threads, gpu_id=0):

	cuda.init()
	evo_ctx = cuda.Device(gpu_id).make_context()
	yolo_ctx = cuda.Device(gpu_id).make_context()

	# Detect machine hardware: CPU and GPUs
	gpu_list, tf_gpu_list, cpu_cores_num = detect_hardware()

	# Pose estimation queues
	joints_queue = Queue()
	joints_to_rec_queues = {0: Queue()}
	joints_to_analyze_queue = Queue()

	import tensorflow as tf
	for gpu_id, tf_gpu in enumerate(tf_gpu_list):
		tf.config.experimental.set_memory_growth(tf_gpu, True)

	processed_joints_queues = {}
	for uik_seq_id in range(cfg.parallel_counting_videos):
		processed_joints_queues[uik_seq_id] = Queue()

	# Action recognition queues
	votes_queue = Queue()
	votes_to_rec_queue = Queue()
	votes_to_analyze_queue = Queue()

	with taken_gpus_lock:
		for gpu_id in gpu_list:
			taken_gpus[gpu_id] = 0

	# Person detector initialization
	from yolov4_trt.utils.yolo_with_plugins import get_input_shape, TrtYOLO
	inp_h, inp_w = get_input_shape(cfg.models.people_detector.model_name)
	yolo_detector = TrtYOLO(
		cfg.models.people_detector.model_name, (inp_h, inp_w), category_num=80,
		model_dir=cfg.models.people_detector.model_dir, cuda_ctx=yolo_ctx
	)

	# Create pose estimation thread
	joints_estimators = []
	for i in range(cfg.je_threads):
		joints_to_rec_thread = JointsEstimator(
			joints_to_rec_queues, joints_to_analyze_queue,
			gpu_id, evo_ctx, queues['api']
		)
		joints_to_rec_thread.setDaemon(True)
		joints_to_rec_thread.start()
		joints_estimators.append(joints_to_rec_thread)

	joints_to_rec_queues[gpu_id].put((None, None, None, None))
	while joints_to_rec_queues[gpu_id].qsize() != 0:
		sleep(0.05)

	# Create action recognition thread
	votes_rec_threads = []
	votes_rec_thread = VotesRecognizer(
		votes_to_rec_queue, votes_to_analyze_queue, gpu_id
	)
	votes_rec_thread.setDaemon(True)
	votes_rec_thread.start()
	votes_rec_threads.append(votes_rec_thread)

	for t in votes_rec_threads:
		votes_to_rec_queue.put((
			None, None, None, None, None,
			None, None, None, None
		))
	while votes_to_analyze_queue.qsize() != 0:
		sleep(0.05)

	# Create threads for preprocessing of input pose estimation data
	joints_transformer_threads = []
	for i in range(cfg.jt_threads):
		joints_transformer_thread = JointsTransformer(
			joints_queue, joints_to_rec_queues
		)
		joints_transformer_thread.setDaemon(True)
		joints_transformer_thread.start()
		joints_transformer_threads.append(joints_transformer_thread)

	# Create threads of detection of ballot box lid intersection
	joints_analyzer_threads = []
	for i in range(cfg.ja_threads):
		joints_analyzer_thread = JointsAnalyzer(
			joints_to_analyze_queue, processed_joints_queues
		)
		joints_analyzer_thread.setDaemon(True)
		joints_analyzer_thread.start()
		joints_analyzer_threads.append(joints_analyzer_thread)

	# Create action recognition preprocessing thread
	votes_transformer_threads = []
	for i in range(cfg.vt_threads):
		votes_transformer_thread = VotesTransformer(
			votes_queue, votes_to_rec_queue, votes_to_analyze_queue
		)
		votes_transformer_thread.setDaemon(True)
		votes_transformer_thread.start()
		votes_transformer_threads.append(votes_transformer_thread)

	# Create action postprocessing thread
	votes_analyzer_threads = []
	for i in range(cfg.va_threads):
		votes_analyzer_thread = VotesAnalyzer(votes_to_analyze_queue)
		votes_analyzer_thread.setDaemon(True)
		votes_analyzer_thread.start()
		votes_analyzer_threads.append(votes_analyzer_thread)

	# Put queues and threads into dictionaries
	queues['joints_transform'] = joints_queue
	queues['joints_to_rec_queues'] = joints_to_rec_queues
	queues['joints_to_analyze'] = joints_to_analyze_queue
	queues['processed_joints'] = processed_joints_queues
	queues['votes_transform'] = votes_queue
	queues['votes_to_rec'] = votes_to_rec_queue
	queues['votes_to_analyze'] = votes_to_analyze_queue

	threads['jt'] = joints_transformer_threads
	threads['je'] = joints_estimators
	threads['ja'] = joints_analyzer_threads
	threads['vt'] = votes_transformer_threads
	threads['vr'] = votes_rec_threads
	threads['va'] = votes_analyzer_threads

	return evo_ctx, yolo_ctx, queues, threads, yolo_detector


def initialize_base_queues(api_token='qwerty'):
	"""
	Create following queues:
		- API queue
		- Ballot boxes recondition queue
		- Sending votes queue
	"""

	api_queue = Queue()
	camera_to_rec_queue = Queue()
	recognized_boxes_queue = Queue()

	# Create API thread
	api_threads = []
	api_thread = API(api_queue, api_token)
	api_thread.setDaemon(True)
	api_thread.start()
	api_threads.append(api_thread)

	# Create sending votes thread
	stop_sending_votes = Event()
	sending_votes_thread = SendVotes(
		stop_sending_votes, cfg.sending_votes_period, api_queue
	)
	sending_votes_thread.start()

	queues = {
		'api': api_queue,
		'find_boxes': camera_to_rec_queue,
		'recognized_boxes': recognized_boxes_queue,
	}
	threads = {
		'api': api_threads,
		'sending_votes': sending_votes_thread,
	}
	events = {
		'sending_votes': stop_sending_votes,
	}

	return queues, threads, events


def check_dirs():
	"""
	- Verify existence of needed directories
	- Verify free disk space
	"""

	# Create the Revisor directories if there are no any
	create_folder(cfg.results_dir)
	create_folder(cfg.votes_vid_dir)
	create_folder(cfg.rejected_votes_vid_dir)
	create_folder(cfg.votes_json_dir)
	create_folder(cfg.stats_dir)
	create_folder(cfg.processed_videos_dir)

	# Create dataset directories if dataset maker mode is enabled
	if cfg.dataset_maker_mode or cfg.reid_dataset_maker_mode:
		create_folder(cfg.dataset_dir)
		create_folder(os.path.join(cfg.dataset_dir, 'videos'))
		create_folder(os.path.join(cfg.dataset_dir, 'json'))
		create_folder(os.path.join(cfg.dataset_dir, 'labelled_videos'))

	# Empty source videos directory if it's enabled in cfg
	if cfg.delete_processed_videos and os.path.isdir(cfg.source_videos_dir):
		source_videos_dirs = os.listdir(cfg.source_videos_dir)
		if len(source_videos_dirs) != 0:
			shutil.rmtree(cfg.source_videos_dir)
	create_folder(cfg.source_videos_dir)

	# Empty temp directory
	if cfg.delete_temp_files and os.path.isdir(cfg.temp_dir):
		temp_dirs = os.listdir(cfg.temp_dir)
		if len(temp_dirs) != 0:
			shutil.rmtree(cfg.temp_dir)
	create_folder(cfg.temp_dir)

	# Check free disk space. Exit system if there are not enough free space.
	free_space_gb = int(shutil.disk_usage("/")[2] // (2 ** 30))
	print('Free disk space: {} GB'.format(free_space_gb))
	continue_processing = True if free_space_gb > cfg.min_free_disk_space else False

	# Create csv with turnout counting statistics
	if not os.path.isfile(cfg.stats_filepath):
		with open(cfg.stats_filepath, 'w') as myFile:
			writer = csv.writer(myFile)
			writer.writerows([[
				'region_uik', 'filename', 'votes', 'rejected_votes',
				'box_intersections', 'uik_total_votes', 'backwards_votes',
				'camera_quality', 'dist_avg', 'width_avg', 'processing_time'
			]])

	if not continue_processing:
		print('Storage is full!')
		sys.exit()


def join_counting(queues, threads):
	"""
	Send "stop" messages to queues and join counting threads
	"""

	queues['joints_transform'].put((
		None, 'STOP', None, None, None,
		None, None, None, None
	))
	queues['joints_transform'].join()

	for q_gpu_id, q in queues['joints_to_rec_queues'].items():
		q.put(('STOP', None, None, None))
		q.join()

	queues['votes_to_rec'].put((
		'STOP', None, None, None, None,
		None, None, None, None
	))
	queues['votes_to_rec'].join()

	queues['joints_to_analyze'].join()
	queues['votes_transform'].join()
	queues['votes_to_rec'].join()
	queues['votes_to_analyze'].join()

	for thr in threads['jt']:
		thr.join()
	for thr in threads['je']:
		thr.join()
	for thr in threads['vr']:
		thr.join()


def join_pipeline(queues, threads, events, join_counting_pipeline=True):
	"""
	Join the Revisor pipeline:
		1. Join counting threads
		2. Join sending votes thread
		3. Join API thread
	"""

	if join_counting_pipeline:
		join_counting(queues, threads)

	events['sending_votes'].set()
	threads['sending_votes'].join()

	queues['api'].join()


def wait_for_free_slot(
		active_threads,
		max_active_threads=cfg.parallel_counting_videos
):
	"""
	Wait until one of created turnout counting threads
	will be free up (will end tasks processing).

	Args:
		active_threads: list of created counting threads
		max_active_threads: max number of threads working simultaneously

	Returns:
		List of created counting threads
	"""

	with processing_counting_tasks_lock:
		active_threads = join_processed_threads(active_threads)

	free_slot = False
	while not free_slot:
		with processing_counting_tasks_lock:
			if len(processing_counting_tasks.keys()) < max_active_threads:
				# free_slot = True
				break
			else:
				active_threads = join_processed_threads(active_threads)
		sleep(3)

	return active_threads


def join_processed_threads(active_threads):
	"""
	Search for turnout counting threads that has been ended tasks processing.
	If any are found, join thread and delete it from the active threads list.

	Args:
		active_threads: list of created counting threads (each element is a thread object)

	Returns:
		List of created counting threads
	"""

	tasks_to_del = []
	objects_to_delete = False
	for task_id, processing in processing_counting_tasks.items():

		# Join turnout counting thread when all its task are done
		if not processing:
			active_threads[task_id].join()
			tasks_to_del.append(task_id)
			objects_to_delete = True

	# Delete task ID:
	# 	- from the active processing tasks
	# 	- from the active threads
	for task_id_to_del in tasks_to_del:
		del processing_counting_tasks[task_id_to_del]
		del active_threads[task_id_to_del]

	if objects_to_delete:
		gc.collect()

	return active_threads


def join_all_counting_threads(active_threads, sleeping_secs=5):
	"""
	Wait until all created turnout counting threads will process all videos
	"""

	# Give some time to initialize turnout counting thread.
	# It's used when we call that function right after creating a thread.
	# sleep(sleeping_secs)

	# Wait for ending processing
	finished = False
	while not finished:
		if len(processing_counting_tasks.keys()) == 0:
			finished = True

		else:
			with processing_counting_tasks_lock:
				active_threads = join_processed_threads(active_threads)
			sleep(sleeping_secs)

	return active_threads


def distribute_task(task_id: int):
	"""
	Distribute a unique queue of analyzed pose coordinates
	data (output of the JointsAnalyzer thread) to the task.
	"""

	all_queue_ids = set(range(0, cfg.parallel_counting_videos))
	with processed_joints_queues_info_lock:

		if processed_joints_queues_info:
			taken_queue_ids = set(processed_joints_queues_info.values())
			available_queue_ids = list(all_queue_ids - taken_queue_ids)
			min_available_queue_id = min(available_queue_ids)

		else:
			min_available_queue_id = 0

		processed_joints_queues_info[task_id] = min_available_queue_id


def clear_gpu_context(ctx):
	if ctx:
		ctx.pop()
		del ctx


if __name__ == '__main__':
	main()
