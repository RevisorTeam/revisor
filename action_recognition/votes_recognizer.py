import json
import os
import numpy as np
import torch.nn as nn
import torch
import warnings
from typing import Tuple


def flatten_over_time(x):
	"""
	LSTM data format
	"""
	return x.reshape(x.shape[0], x.shape[1] * x.shape[2])


def get_joints_lengths(points: np.array, connected_joints: list, n_joints: int = 17):
	bones_lengths = np.zeros(n_joints)

	for i, pair in enumerate(connected_joints):
		length = np.linalg.norm(points[pair[1]]-points[[pair[0]]])
		bones_lengths[i] = length
	return bones_lengths


def parse_data(data, use_shift=False):
	inward = [(10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 13),
			  (16, 14), (14, 12), (11, 5), (12, 6), (5, 0), (6, 0),
			  (1, 0), (2, 0), (1, 3), (2, 4), (12, 11)]

	frames = data["joint_coords"]
	orientations = data["voter_orientation"]
	scores = data["joint_confidences"]

	start_frame = data.get("voting_start_frame", 0)
	end_frame = data.get("voting_end_frame", 0)

	keys = list(frames.keys())
	keys = [int(key) for key in keys]

	target_points = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
					 'left_shoulder', 'right_shoulder', 'left_elbow',
					 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
					 'right_hip', 'left_knee', 'right_knee', 'left_foot',
					 'right_foot']

	min_frame, max_frame = min(keys), max(keys)

	n_joints = len(target_points)
	n_features = 18
	X = np.zeros((max_frame - min_frame + 1, n_features, 4))
	X_score = np.zeros((max_frame - min_frame + 1, n_joints))

	last_points, last_scores = np.zeros((n_features, 2)), np.zeros(n_joints)

	for k, frame_num in zip(range(len(X)), range(min_frame, max_frame + 1)):
		current_frame = frames.get(str(frame_num), dict())
		current_score = scores.get(str(frame_num), dict())

		cur_points = np.zeros((n_features, 2))
		cur_scores = np.zeros(n_joints)

		correct_frame = True

		for i, point_name in enumerate(target_points):
			if point_name not in current_frame:
				correct_frame = False
				break
			else:
				point = current_frame.get(point_name)
				score = current_score.get(point_name)

				cur_points[i] = point
				cur_scores[i] = score

		angle = orientations.get(str(frame_num), dict()).get("angle", 0)
		angle /= 360
		cur_points[n_features - 1, 0] = angle

		if correct_frame:
			X[k, :, :2] = cur_points
			X[k, :n_joints, 3] = get_joints_lengths(cur_points, inward)
			X_score[k] = cur_scores

			last_points = cur_points
			last_scores = cur_scores
		else:
			X[k, :, :2] = last_points
			X[k, :n_joints, 3] = get_joints_lengths(last_points, inward)
			X_score[k] = last_scores

	shift = start_frame - min_frame
	x_size = end_frame - start_frame + 1

	if use_shift:
		X = X[shift: shift + x_size]
		X_score = X_score[shift: shift + x_size]

	cap_centroid = data["cap_centroid"][next(iter(data["cap_centroid"]))]
	bx, by = cap_centroid["x"], cap_centroid["y"]
	cap_bbox = data["cap_bbox"][next(iter(data["cap_bbox"]))]

	X[:, :n_features - 1, 0] -= bx
	X[:, :n_features - 1, 1] -= by

	X[:, :n_joints, 2] = X_score

	X[:, :n_features - 1, :2] -= X[:, :n_features - 1, :2].mean(axis=0).mean(axis=0)
	X[:, :n_features - 1, :2] /= X[:, :n_features - 1, :2].mean(axis=0).std(axis=0)

	X[:, :n_features - 1, 3] -= X[:, :n_features - 1, 3].mean(axis=0).mean(axis=0)
	X[:, :n_features - 1, 3] /= X[:, :n_features - 1, 3].mean(axis=0).std(axis=0)

	box_type = data["voted_ballot_box_type"]
	X[:, n_features - 1, 1] = box_type

	return X


class lstm(nn.Module):
	def __init__(self, input_size, bidirectional=False):
		super(lstm, self).__init__()
		self.lstm = nn.LSTM(input_size=input_size, hidden_size=256, num_layers=1, batch_first=True, dropout=0.5,
							bidirectional=bidirectional)
		self.fc1 = nn.Linear(in_features=256, out_features=64)
		self.fc2 = nn.Linear(in_features=64, out_features=2)
		self.act = nn.ReLU(inplace=True)
		self.dropout = nn.Dropout(p=0.5)
		self.final = nn.Softmax()

	def forward(self, x):
		output, _status = self.lstm(x)
		output = output[:, -1, :]
		output = self.fc1(output)
		output = self.act(output)
		output = self.dropout(output)
		output = self.fc2(output)
		output = self.final(output)
		return output


class VotingModel:
	def __init__(
			self,
			checkpoint: str,
			cut_size: int = 631,
			input_shape: int = 72,
			threshold: float = 0.5,
			window_size=3
	):
		self.threshold = threshold
		self.cut_size = cut_size
		self.checkpoint = checkpoint
		self.window_size = window_size
		self.input_shape = input_shape
		self.model = None
		self.__load_model()
		self.__init_environment()

	def predict(self, data: dict) -> Tuple[int, float]:

		x = parse_data(data)
		x = self.__preprocess_x(x)
		x = flatten_over_time(x)
		x = torch.cuda.FloatTensor(x)
		with torch.no_grad():
			output = self.model(x[None, ...])

		pos_class_prob = output.data[:, 1].cpu().numpy()[0]
		predicted_label = 1 if pos_class_prob > self.threshold else 0

		return predicted_label, pos_class_prob

	@staticmethod
	def __init_environment() -> None:
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.enabled = True

		warnings.simplefilter("ignore")
		os.environ["PYTHONWARNINGS"] = "ignore"

	def __load_model(self) -> None:
		self.model = lstm(self.input_shape)

		model_dict = torch.load(self.checkpoint)
		self.model.load_state_dict(model_dict['state_dict'])
		self.model = self.model.cuda(device=0)
		self.model.eval()

	def __preprocess_x(self, x: np.array):
		if self.window_size:
			x = np.array(
				[np.concatenate((x[max(0, i - self.window_size): i + 1, :-1].mean(axis=0),
								 time_step[-1:]))
				 for i, time_step in enumerate(x)]
			)

		x_swapped = np.swapaxes(x, 1, 2).reshape(-1, 4, x.shape[1])
		x_pad = np.full((self.cut_size, x_swapped.shape[1], x_swapped.shape[2]), fill_value=-999)
		x_pad[-x.shape[0]:, :, :] = x_swapped

		return x_pad


def main():
	model = VotingModel(
		checkpoint='weights/votes_recognizer.pt',
		threshold=0.743
	)
	data = json.load(open('json_examples/5_928_1_rejected2.json', 'rb'))
	print(model.predict(data))


if __name__ == "__main__":
	main()
