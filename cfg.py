class Config(object):
	def __init__(self, config_dict):
		for key, val in config_dict.items():
			self.__setattr__(key, val)


models = Config({

	# Ballot boxes detection model (Yolact)
	'boxes': Config({
		'config': 'yolact_im700_config_v3',
		'weights': 'yolact/weights/boxes_finder.pth',
	}),

	# Ballot boxes lid detection model (QueryInst)
	'boxes_cap': Config({
		'config': 'QueryInst/configs/queryinst_cap_V2.py',
		'weights': 'QueryInst/weights/lid_finder.pth',
	}),

	# Person detection model (YOLOv4 - TensorRT engine)
	'people_detector': Config({
		'model_name': 'yolov4-608',
		'model_dir': 'yolov4_trt/yolo',
	}),

	# Pose estimation model (EvoPose2D - TensorRT engine)
	'pose': Config({

		# EvoPose2D batch size
		'batch_size': 40,

		# EvoPose2D M
		'model_path': 'evopose2d/models/evopose2d_M_batch40.trt',
		'input_shape': [384, 288, 3],
		'output_shape': [192, 144, 17],

	}),

	# Votes classification model (custom torch LSTM model)
	'action': Config({
		'weights': 'action_recognition/weights/votes_recognizer.pt',
		'conf_threshold': 0.3
	})
})


cfg = Config({

	# Models config
	'models': models,

	# --------------------------------------------------------------------------------
	# Main variables
	# --------------------------------------------------------------------------------
	# Path to directory with source videos
	'source_videos_dir': 'data/source_videos',

	# Path to the CSV with source directories to process
	'target_dirs': 'target_dirs.csv',

	# Path to JSON with polling stations docs
	'stations_path': 'data/stations_demo.json',

	# Temp directory path
	'temp_dir': 'data/temp',

	# Total amount of pixels inside frame. Declared performance
	# guaranteed only with target_pixels=307200 (640 * 480).
	# Videos will be resized to this amount of pixels during processing.
	'target_pixels': 640 * 480,

	# Whether to delete temp directory
	'delete_temp_files': False,

	# Whether to delete source videos dir after processing.
	'delete_processed_videos': False,

	# --------------------------------------------------------------------------------
	# Ballot boxes detection module variables
	# ballot_boxes_finder.py
	# --------------------------------------------------------------------------------
	# Total number of frames to analyze
	'count_analyzed_frames': 200,

	# Number of seconds to analyze (from the beginning)
	'find_boxes_video_length': 60,

	# Number of analyzing frames per second
	'analyzed_frames_per_sec': 5,

	# Number of frames processing simultaneously (in the Yolact)
	'parallel_frames': 1,

	# Bounding box intersection threshold. If intersection rate is larger,
	# box will be deleted
	'intersection_threshold': 0.7,

	# Upscale coefficients
	'bbox_upscale_k': 0.4,
	'cap_bbox_upscale_k': 0.2,
	'cap_rotated_bbox_upscale_k': 0.4,

	'min_max_mask_area': 0.2,
	'min_mask_ratio': 0.15,
	'side_min': 20,

	# Show cv2 window with ballot boxes tracking
	'visualize_object_tracking': False,

	# Show window with resulting recognized image
	'visualize_bboxes_quality': False,

	# Show windows with detected ballot boxes lids
	'visualize_rec_cap_masks': False,

	# Save image with detected boxes and camera docs
	'save_bboxes_quality_image': True,

	# Save json with ballot boxes docs
	'save_json_output': True,

	# Delete temp video (located in temp_dir)
	'delete_temp_video': False,

	# Save video with segmentated ballot boxes
	'save_rec_video': True,


	# --------------------------------------------------------------------------------
	# Turnout counting module variables
	# revisor.py
	# --------------------------------------------------------------------------------

	##########
	# Main variables
	##########
	# Target video frame rate. Other frames will be skipped
	'target_fps': 8,

	# List of allowed video formats. All videos must have constant fps
	'allowed_video_formats': ['mp4', 'avi'],

	# Number of source video seconds to store in RAM
	'frames_memory_seconds': 250,

	# Number of stations to process simultaneously
	'parallel_counting_videos': 1,

	# Min threshold of free disk space (Gb) to launch the Revisor
	'min_free_disk_space': 5,

	# Refreshing time of the global votes list
	'sending_votes_period': 30,

	# Polling station working hours (including start and end hour).
	# Used only in API mode.
	'approved_hours': {'start': 8, 'end': 19},

	# Delete regular ballot boxes when there is 'koib' (electronic ballot box)
	# on the station
	'delete_ballot_boxes': False,

	# Number of seconds when tracker will be considered to be out of frame
	'vanish_seconds': 4,

	# Coefficient to find radius of the stopped area circle
	'ballot_box_zone_k': 1.2,

	# Min number of frames person should be near
	# a ballot box to be considered as stopped
	'min_frames_to_stop': 19,

	# Every X frame is taken for determination of stopping near a box
	'centroids_analyzing_frame': 3,

	# Size of the tracker centroid circle during determination
	# of stopping near a box stage
	'avg_voter_centroid_k': 0.13,

	# Min and max near box tracker frames threshold
	# to be sent to the pose estimation
	'min_near_box_frames': 12,
	'max_near_box_frames': 800,

	# Min and max number of tracker's seconds / frames when hand must be inside lid
	# to be sent to action recognition
	'min_voting_time': 0.5,
	'max_voting_frames': 631,

	# Shift in seconds from start and end of the action (for output samples)
	'vote_shift_time': 2,

	# Joints names which are used for determination of hands
	# intersection ballot box lid zone.
	'target_joints': [
		'left_elbow', 'right_elbow',
		'left_wrist', 'right_wrist',
	],

	##########
	# Selecting videos to detect boxes.
	# One of three following bools may be True.
	# If all set to false - random video will be picked up.
	##########
	# Always get the earliest available video for boxes detection
	'get_first_vid_for_boxes_rec': True,

	# Get video for boxes detection from the target starting time.
	# If there is no video with the same starting time - polling station will be skipped.
	'get_target_time_for_boxes_rec': False,
	'boxes_rec_target_time': {'hour': 12, 'minute': 0},

	# Get video for boxes detection from the target time period (starting and ending hours).
	# If there is no video with the same starting time - random video will be taken.
	'get_timing_for_boxes_rec': False,
	'boxes_approved_vid_times': {'start': 12, 'end': 13},

	##########
	# Person orientation
	##########
	# Direction names of the person orientation (relative to camera)
	'direction_names': [
		'back', 'back_right',
		'right', 'front_right',
		'front', 'front_left',
		'left', 'back_left'
	],

	# Direction angles (from 0.0 to 360.0) of the person orientation
	'direction_angles': {
		'back': [67.5, 112.5],
		'back_right': [112.5, 157.5],
		'right': [157.5, 202.5],
		'front_right': [202.5, 247.5],
		'front': [247.5, 292.5],
		'front_left': [292.5, 337.5],
		'left': [337.5, 22.5],
		'back_left': [22.5, 67.5]
	},

	# Orientation names of the positive oY axis
	'abs_side_names': [
		'left', 'top_left', 'top',
		'top_right', 'right'
	],

	##########
	# Datasets gathering
	##########
	# Unlabelled action recognition dataset gathering mode.
	# Collecting all samples when person's hand intersected
	# ballot box lid zone more than min_voting_time seconds.
	'dataset_maker_mode': False,

	# Unlabelled ReID dataset gathering mode.
	# Collecting all visible samples in the video.
	# (ballot boxes aren't used)
	'reid_dataset_maker_mode': False,

	# Whether to save videos with visualized data (bboxes, poses, etc)
	'save_labelled_videos': False,

	# Output dataset directory path
	'dataset_dir': 'path_to_output_dataset_dir',

	##########
	# Logging and visualization
	##########
	# Log recognized actions
	'log_votes': True,

	# Log pose estimation
	'log_pose_processing': False,

	# Log pose estimation queue size
	'log_pose_queue_sizes': False,

	# Log API
	'log_api': False,

	# Show cv2 window with processing video docs.
	'show_recognitions': False,

	# Visualize data (bboxes, poses, etc) on output video
	# (in labelled_videos dir).
	'draw_skeletons': False,

	# Show cv2 window with votes actions
	'show_votes_window': False,

	# Show cv2 window with other actions
	'show_rej_votes_window': False,

	##########
	# The Revisor results
	##########
	# Save video with processing video docs
	'save_processing_frames': False,

	# Whether to save votes videos
	'save_votes_vid': False,

	# Whether to save denied actions videos
	'save_rejected_votes_vid': False,

	# Whether to save votes jsons
	'save_votes_json': False,

	# Whether to save denied actions jsons
	'save_rejected_json': False,

	# The Revisor results directory
	'results_dir': 'data/revisor_results',

	# Statistics csv with turnout counting results
	'stats_filepath': 'data/revisor_results/stats/stats.csv',

	# Whether to save csv with voting actions docs
	# to results_dir/stats/votes.csv.
	# CSV contains following:
	# region_num, station_num, local datetime, vote confidence
	'save_votes_times': True,

	# Directory which contains votes.csv
	'stats_dir': 'data/revisor_results/stats',

	# Directory with votes videos
	'votes_vid_dir': 'data/revisor_results/videos/votes',

	# Directory with denied actions videos
	'rejected_votes_vid_dir': 'data/revisor_results/videos/rejected_votes',

	# Directory with votes and denied actions jsons
	'votes_json_dir': 'data/revisor_results/json',

	# Directory with processed videos names CSVs
	'processed_videos_dir': 'data/revisor_results/processed_videos',

	# Processed polling stations CSV path
	'processed_videos_csv': 'data/revisor_results/processed_videos/processed_uiks.csv',

	##########
	# Threads
	##########

	# CPU + GPU threads:

	# Number of pose estimation preprocessing threads
	'jt_threads': 1,

	# Number of pose estimation threads
	'je_threads': 1,

	# Number of votes recognition threads
	'vr_threads': 1,

	# CPU only threads:

	# Number of lid intersection determination threads
	'ja_threads': 1,

	# Number of action recognition preprocessing threads
	'vt_threads': 1,

	# Number of action recognition postprocessing threads
	'va_threads': 1,

	##########
	# API - not revealed in the current version of the Revisor
	##########

	# What pipeline to use: 'local' or 'api'.
	'data_source': 'local',

	# Whether to send data to the API
	'send_data': False,

})
