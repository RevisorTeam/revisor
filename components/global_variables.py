from threading import Lock


processing_boxes_tasks = []
processing_counting_tasks = {}
taken_gpus = {}
processed_joints_queues_info = {}		# == uiks_processed_joints_queues

vote_times = {}
counters = {}
frames_memory = {}
evo_batches = {}

ballot_boxes_data, cap_polygons_data = {}, {}

vote_times_lock = Lock()
processing_boxes_tasks_lock = Lock()
processing_counting_tasks_lock = Lock()
frames_memory_lock = Lock()
counters_lock = Lock()
taken_gpus_lock = Lock()
saving_lock = Lock()
processed_joints_queues_info_lock = Lock()
videos_counted_lock = Lock()
evo_batches_lock = Lock()
poses_queue_lock = Lock()
ballot_boxes_lock = Lock()
