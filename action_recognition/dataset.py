import numpy as np
import pickle

from sklearn.utils.class_weight import compute_class_weight
from utils import flatten_over_time
from torch.utils.data import Dataset


def shift(x: np.array, p=0.2):
    if np.random.randint(0, 100) < p*100:
        shift_size = max(x.shape[0] // 10, 1)
        x = x[shift_size:, :, :]
    return x


def aug_coords(x: np.array, p=0.2):
    if np.random.randint(0, 100) < p * 100:
        n_frames = x.shape[0]
        frames = np.random.choice(np.arange(0, n_frames), size=max(x.shape[0] // 10, 1))

        for frame in frames:
            num_coords_for_aug = np.random.randint(0, 9)
            coords_for_aug = np.random.choice(np.arange(0, 17), size=num_coords_for_aug)

            for coord in coords_for_aug:
                for ax in [0, 1]:
                    min_value, max_value = x[:, coord, ax].min(), x[:, coord, ax].max()
                    scope = max_value - min_value
                    scope /= np.random.randint(10, 50)

    return x


def swap_frames(x: np.array, p=0.2):
    if np.random.randint(0, 100) < p * 100:
        n_frames = x.shape[0]
        swap_index = np.random.choice(np.arange(2, n_frames-2))

        direction = np.random.choice([-1, 1])

        x[swap_index+direction, :, :], x[swap_index, :, :] = x[swap_index, :, :], x[swap_index+direction, :, :]

    return x


class Feeder(Dataset):
    def __init__(self, annotations_paths, inplace_preprocess, window_size=3, cut_size=602, random_shift=False,
                 coords_aug=False, random_swap=False, debug=False, test=False):
        """
        :param window_size: The length of the output sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """
        self.test = test

        labels_mapping_train = {0: [0, 10], 1: [1], 2: [2, 3, 4, 5, 6, 7, 8, 9]}
        labels_mapping_test = {0: [0, 6, 7, 10], 1: [1, 2, 4, 3, 5, 8, 9]}

        labels_mapping = labels_mapping_test if self.test else labels_mapping_train

        self.labels_mapping = {class_id: binary_class for binary_class in labels_mapping.keys()
                               for class_id in labels_mapping[binary_class]}

        self.debug = debug
        self.annotations = pickle.load(open(annotations_paths, 'rb'))
        self.window_size = window_size
        self.load_data()
        self.inplace_preprocess = inplace_preprocess
        self.cut_size = cut_size
        self.random_shift = random_shift
        self.coords_aug = coords_aug
        self.random_swap = random_swap

    def load_data(self):
        self.paths = [i['path'] for i in self.annotations]
        self.labels = [self.labels_mapping[i['label']] for i in self.annotations]
        self.or_labels = [i['label'] for i in self.annotations]

        reduced = np.array([i for i, label in enumerate(self.labels) if label != 2])

        self.paths = list(np.array(self.paths)[reduced]) if not self.test else self.paths
        self.labels = list(np.array(self.labels)[reduced]) if not self.test else self.labels

        if self.debug:
            self.paths = self.paths[0:8000]
            self.labels = self.labels[0:8000]

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        return self

    def process_x(self, x: np.array):
        # x[:, 17, 1] = 0
        if self.window_size:
            x = np.array(
                [np.concatenate((x[max(0, i - self.window_size): i + 1, :-1].mean(axis=0),
                                 time_step[-1:]))
                 for i, time_step in enumerate(x)]
            )

        # x = x[-self.cut_size:]
        if self.random_shift:
            x = shift(x)
        if self.coords_aug:
            x = aug_coords(x)
        if self.random_swap:
            x = swap_frames(x)

        x_swapped = np.swapaxes(x, 1, 2).reshape(-1, 4, x.shape[1])
        x_pad = np.full((self.cut_size, x_swapped.shape[1], x_swapped.shape[2]), fill_value=-999)
        x_pad[-x.shape[0]:, :, :] = x_swapped

        return x_pad

    def __getitem__(self, index):
        data_numpy = pickle.load(open(self.paths[index], 'rb'))
        if self.inplace_preprocess:
            data_numpy = self.process_x(data_numpy)
        label = self.labels[index]
        data_numpy = flatten_over_time(np.array(data_numpy))
        path = self.paths[index]
        or_label = self.or_labels[index]

        if self.test:
            return data_numpy, label, path, or_label

        return data_numpy, label

    def compute_class_weights(self):
        return compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels)
