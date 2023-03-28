from utils import get_bbox_params


class TrackingVoter:
	def __init__(self, obj_id, bbox):

		# Camera person ID (not equal to voter turnout number)
		self.voter_id = obj_id

		# Person bbox coordinates on the current frame
		self.bbox = {
			'x1': bbox[0], 'y1': bbox[1],
			'x2': bbox[2], 'y2': bbox[3]
		}

		# Centroid coordinates, width and width of the person bbox on the current frame
		self.centroid, self.bbox_width, self.bbox_height = get_bbox_params(
			self.bbox['x1'], self.bbox['y1'],
			self.bbox['x2'], self.bbox['y2'],
			out_type='int'
		)

		# List of all person's centroids
		self.centroids = [self.centroid]

		# Dict with all tracker's bboxes. Mapping: {frame_id: bbox, ...}
		self.bboxes = {}

		# Person joints on the current frame
		self.current_joints = {}

		# Last frame_id with estimated joints
		self.last_joints_frame_id = None

		# Dict with joints on all frames: {frame_id: joints, ...}
		self.joints = {}

		# Dict with confidences of predicted joints: {frame_id: confidences, ...}
		self.joints_confidences = {}

		# Whether person is visible on the current frame
		self.visible = False

		# First frame_id when person appeared in the camera
		self.first_visible_frame = None

		# Last frame_id when person was visible in the camera
		self.last_visible_frame = None

		# Whether person's hands was inside a ballot box lid zone
		self.intersected_cap = False

		# Frame_ids when person's hands was inside a lid zone
		self.intersected_ballot_box = []

		# Whether person stopped near ballot box
		self.stopped_near_box = False

		# Frame_ids when person was near ballot box
		self.near_box_frames = []

		# Frame_ids when person stopped near ballot box
		self.stopped_near_box_frames = []

		# Person is currently being processed by pose estimation thread
		self.recognizing_joints = False

		# Person is has been processed by pose estimation thread
		self.joints_processed = False

		# Frame_id of first intersection of lid zone by person hands
		self.voting_start_frame_id = None

		# Frame_id of last intersection of lid zone by person hands
		self.voting_end_frame_id = None
		self.voting_end_raw_frame_id = None

		# Number of hands intersection of each ballot box lid
		self.box_intersections_num = {}

		# Voted ballot box id
		self.voted_ballot_box = None
