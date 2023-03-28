from cfg import cfg
from utils import get_camera_info, read_csv, read_json, find_station_by_cam_id

import random
from datetime import timedelta
from datetime import datetime
import os


def get_local_videos() -> dict:
	"""
	Parse cameras listed in target_dirs.csv.
	For each camera (target directory) collect metadata of containing videos.

	Path to the CSV file is defined in "target_dirs" variable
	inside the config file (cfg.py).

	Return:
		Function "get_local_videos" returns "tasks" dictionary, consisting of:

		Each key of "tasks" dict is a task_id or source_dir_id (0...N)

		'region_number' - polling stating region number
		'station_number' - polling station number
		'cam_number' - camera number on the station
		'cam_id' - camera unique identifier (str)
		'box_type' - ballot box type
		'target_dir' - directory name with camera videos (not a path)
		'videos_dir' - path to a directory with camera videos
		'videos' - dict with a camera videos. Each key is a video_id (0...K)
			'path' - path to a video
			'filename' - video filename
			'start_time' - starting local datetime of a video
			'end_time' - ending local datetime of a video
			'start_hour_num' - starting hour of a video
			'end_hour_num' - ending hour of a video
	"""

	tasks = {}

	target_dirs = read_csv(cfg.target_dirs)
	processed_dirs = read_csv(cfg.processed_videos_csv)
	stations = read_json(cfg.stations_path)

	uiks_num = 0
	for uik_id, target_dir in enumerate(target_dirs):

		# If current target dir has already been processed, skip it
		if target_dir in processed_dirs:
			continue

		# Process parallel_counting_videos simultaneously
		if uiks_num >= cfg.parallel_counting_videos:
			break

		videos_dir = os.path.join(cfg.source_videos_dir, target_dir)
		region_num, uik_num, camera_num, camera_id = get_camera_info(target_dir)

		# Skip dir if stations json file doesn't have target station inside
		uik_info = find_station_by_cam_id(camera_id, stations)
		if not uik_info:
			continue

		timezone_offset = timedelta(minutes=uik_info['timezone_offset_minutes'])

		processed_files_csv = os.path.join(
			cfg.processed_videos_dir,
			'{}_{}_{}_processed_videos.csv'.format(region_num, uik_num, camera_num)
		)
		processed_files = read_csv(processed_files_csv)

		# If there are no videos in target_dir, skip it
		files_list = os.listdir(videos_dir)
		if not files_list:
			continue

		videos_to_process = {}
		for file_id, file in enumerate(files_list):

			# Video must have allowed file format
			file_format = file.split('.')[-1]
			if file_format not in cfg.allowed_video_formats:
				continue

			# Process videos that have not been processed before
			if file in processed_files:
				continue

			epochs = file.split('_')[3]
			start_hour_num = file.split('_')[0]
			end_hour_num = file.split('_')[1].split('-')[-1]
			start_epoch = int(epochs.split('-')[0])
			end_epoch = int(epochs.split('-')[-1].split('.')[0])

			# Parse video's local starting and ending datetimes
			start_local_datetime = datetime.utcfromtimestamp(start_epoch) + timezone_offset
			end_local_datetime = datetime.utcfromtimestamp(end_epoch) + timezone_offset

			video_path = os.path.join(videos_dir, file)
			videos_to_process[file_id] = {
				'path': video_path,
				'filename': file,
				'start_time': start_local_datetime,
				'end_time': end_local_datetime,
				'start_hour_num': int(start_hour_num),
				'end_hour_num': int(end_hour_num)
			}

		# In test stations.json "koib_keg" key stands for ballot box type:
		# 	* 0: ballot_box - regular ballot box
		# 	* 1: koib - electronic ballot box
		# 	* 2: keg - another version of electronic ballot box (not presented here)
		tasks[uik_id] = {
			'uik_id': uik_id,
			'region_number': region_num,
			'station_number': uik_num,
			'cam_number': camera_num,
			'cam_id': camera_id,
			'box_type': uik_info['koib_keg'] if 'koib_keg' in uik_info.keys() else None,
			'target_dir': target_dir,
			'videos_dir': videos_dir
		}
		tasks[uik_id]['videos'] = videos_to_process
		uiks_num += 1

	return tasks


def get_boxes_rec_vid(videos_to_process) -> str:
	"""
	Pick a file for ballot boxes detection (in 'local' mode only)
	"""

	boxes_rec_filename = ''

	if cfg.reid_dataset_maker_mode:
		return boxes_rec_filename

	for file_id, file_info in videos_to_process.items():

		boxes_rec_filename = file_info['path']

		# Get video starting in the target hour and minute
		if cfg.get_target_time_for_boxes_rec:
			if file_info['start_time'].hour == cfg.boxes_rec_target_time['hour'] and \
				file_info['start_time'].minute == cfg.boxes_rec_target_time['minute']:
				return boxes_rec_filename

		# Get the earliest camera video (first video)
		if cfg.get_first_vid_for_boxes_rec:
			return boxes_rec_filename

		# Get video starting in the target hour period
		if cfg.get_timing_for_boxes_rec:
			start_hour = int(file_info['start_hour_num'])
			if cfg.boxes_approved_vid_times['start'] <= start_hour <= cfg.boxes_approved_vid_times['end']:
				return boxes_rec_filename

	# Get random video if a target video wasn't found
	if not boxes_rec_filename:
		rand_file_id = random.choice(list(videos_to_process.keys()))
		boxes_rec_filename = videos_to_process[rand_file_id]['path']

	return boxes_rec_filename
