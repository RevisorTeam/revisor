import os
import pandas as pd
import numpy as np
import pickle
import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
import dataset as ds
import torch.nn.functional as F

from torch.optim.lr_scheduler import _LRScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, fbeta_score,
    precision_score, recall_score,
    accuracy_score, matthews_corrcoef,
    confusion_matrix
)
from typing import List, Tuple
from loguru import logger
import warnings


def get_joints_lengths(points: np.array, connected_joints: list, n_joints: int = 17):
    bones_lengths = np.zeros(n_joints)

    for i, pair in enumerate(connected_joints):
        length = np.linalg.norm(points[pair[1]]-points[[pair[0]]])
        bones_lengths[i] = length
    return bones_lengths


def parse_data(data, use_shift=True):
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

    box_type = data["uik_boxes_type"]
    X[:, n_features - 1, 1] = box_type

    return X


def create_dataset_paths(data: pd.DataFrame, ds_path: str) -> pd.DataFrame:
    ds_files = os.listdir(ds_path)
    res = None

    for i, row in data.iterrows():
        sub_df = [('_'.join(i.split('_')[:3]), os.path.join(ds_path, i), i.replace('.json', '').split('_')[3], row['box_type'], row['selection'])
                  for i in ds_files if i.startswith(row['reg_uik'])]
        sub_df = pd.DataFrame(sub_df, columns=['uik_id', 'path', 'sample_id', 'classes', 'selection'])

        if res is None:
            res = sub_df
        else:
            res = pd.concat([res, sub_df])

    return res


def split_dataset(df: pd.DataFrame, split_ratio: List[float], needed_cols: List[str], target_col: str,
                  n_bins: int = 10) -> Tuple[List, List, List]:
    for col in needed_cols:
        df[col] = list(map(lambda x: float(x.replace(',', '.')), df[col].values))
        dist, ranges = np.histogram(df[col], bins=n_bins)

        # calculating frequency in bins
        df[f"{col}_bin"] = df.apply(lambda row: np.digitize(row[col], ranges), axis=1)
        df[f"{col}_freq"] = df.apply(lambda row: len(df[df[f"{col}_bin"] == row[f"{col}_bin"]]), axis=1)

        df = df[df[f"{col}_bin"]>2]
        df = df[df['avg_normalized_width_k'] > 0.35]

    # stratifying by feature distribution
    X_train, X_test = train_test_split(df, stratify=df[[f"{col}_freq" for col in needed_cols]], train_size=split_ratio[0])
    X_test, X_val = train_test_split(X_test, stratify=X_test[[f"{col}_freq" for col in needed_cols]], test_size=split_ratio[1])

    X_train['selection'] = 'train'
    X_test['selection'] = 'test'
    X_val['selection'] = 'val'

    res = pd.concat([X_train, X_test, X_val])
    return res[[target_col, 'box_type', 'selection']]


def split_by_bbox_types(df: pd.DataFrame):
    res = pd.DataFrame()

    for i, group in df.groupby('box_type'):
        splitted_df = split_dataset(group, [0.6, 0.5], ['avg_normalized_width_k'], 'reg_uik', 200)
        res = pd.concat([res, splitted_df])

    paths = create_dataset_paths(res, 'voting_dataset_v2')
    paths.to_csv('paths.csv', index=False)

    return paths


def load_pickle(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)

    return data


def save_pickle(path, data):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)


def flatten_over_time(x):
    return x.reshape(x.shape[0], x.shape[1] * x.shape[2])


class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., alpha=1.):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input_tensor, target_tensor):
        ce_loss = torch.nn.functional.cross_entropy(input_tensor, target_tensor,
                                                    reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()  # mean over the batch
        return focal_loss


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


class TrainingLoop:
    def __init__(self, window_size: int = 3, cut_size: int = 224, batch_size: int = 1536, base_lr: float = 0.005,
                 optimizer: str = 'Adam', epochs: int = 150, steps: list = [45, 55, 70, 80, 110], prepare_X: bool = False,
                 warm_up_epoch: int = 0, model_type: str = None, inplace_preprocess: bool = True, loss: str = 'bce',
                 reweight=True, checkpoint=None):
        self.window_size = window_size
        self.cut_size = cut_size
        self.model_type = model_type
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.optimizer = optimizer
        self.epochs = epochs
        self.steps = steps
        self.warm_up_epoch = warm_up_epoch
        self.loss = loss
        self.reweight = reweight
        self.checkpoint = checkpoint
        self.inplace_preprocess = inplace_preprocess

        if prepare_X:
            self.prepare_X_data(True)
        else:
            self.load_data()

        self.model = self.get_model()
        self.init_environment()

    @staticmethod
    def init_environment():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    def train(self, output_folder: str = 'checkpoints', cuda: bool = True):
        option_name = f"{self.model_type}_{self.loss}_reweight_{self.reweight}_centroid_{self.cut_size}_" \
                      f"{('ma_' + str(self.window_size)) if self.window_size else ''}"

        output_folder = os.path.join(output_folder, option_name)
        os.makedirs(output_folder, exist_ok=True)

        logger.debug(f"{self.class_weights=}")
        class_weight = torch.FloatTensor(self.class_weights).cuda() if self.reweight else None

        if self.loss == 'bce':
            criterion = torch.nn.BCELoss(weight=None)
        else:
            criterion = FocalLoss(weight=None)

        if self.checkpoint:
            model = torch.load(self.checkpoint)
            self.model.load_state_dict(model['state_dict'])

        if cuda:
            self.model = self.model.cuda()
            criterion = criterion.cuda()

        self.load_optimizer()

        best_score = float('-inf')

        for epoch in range(self.epochs):
            logger.debug(f"Epoch {epoch}/{self.epochs-1}")
            self.adjust_learning_rate(epoch)
            for phase in ['train', 'val']:
                if phase == 'train':
                    dataloader = self.data_loader['train']
                    self.model.train()
                else:
                    dataloader = self.data_loader['val']
                    self.model.eval()

                running_loss = list()
                y_pred = list()
                y_test = list()

                for batch_idx, (data, label) in enumerate(tqdm.tqdm(dataloader)):
                    if cuda:
                        data = data.float().cuda()
                        label = label.float().cuda()

                    # logger.debug(f"{label=}")

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        output = self.model(data)

                        if isinstance(output, tuple):
                            output, l1 = output
                            l1 = l1.mean()
                        else:
                            l1 = 0

                        # logger.debug(f"{output=}")
                        # logger.debug([output.shape, label.shape])
                        loss = criterion(output, label.reshape(-1, 1)) + l1

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    running_loss.append(loss.data.item())
                    predict_label = torch.round(output.data)
                    # logger.debug(f"{predict_label=}")
                    y_pred += [i[0] for i in predict_label.data.int().cpu().detach().tolist()]
                    y_test += label.int().cpu().detach().tolist()

                fhalf_score = calc_metrics(y_test, y_pred, phase=phase, test=False)

                self.lr = self.optimizer.param_groups[0]['lr']

                logger.debug(f"loss value: {np.mean(running_loss)} | lr: {self.lr}")
                logger.debug(f"best val score = {best_score}")

                if phase == 'val' and fhalf_score > best_score:
                    best_score = fhalf_score

                    torch.save({'state_dict': self.model.state_dict(), 'best_fscore': best_score},
                               os.path.join(output_folder, f"model_{epoch}.pt"))

                    torch.save({'state_dict': self.model.state_dict()},
                               os.path.join(output_folder, f"model_best.pt"))

    def adjust_learning_rate(self, epoch):
        if epoch < self.warm_up_epoch:
            lr = self.base_lr * (epoch + 1) / self.warm_up_epoch
        else:
            lr = self.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(self.steps)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def load_optimizer(self):
        if self.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.base_lr,
                momentum=0.99,
                nesterov=True,
                weight_decay=1e-4)

        elif self.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.base_lr,
                weight_decay=1e-4)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.steps, gamma=0.2)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)

        logger.debug('using warm up, epoch: {}'.format(self.warm_up_epoch))

    def load_data(self):
        self.data_loader = dict()

        dataset=ds.Feeder(annotations_paths='train_annotations.pkl', inplace_preprocess=self.inplace_preprocess,
                              cut_size=self.cut_size, random_shift=True, random_swap=True, coords_aug=True)
        dataset.load_data()
        self.class_weights = dataset.compute_class_weights()

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=ds.Feeder(annotations_paths='test_annotations.pkl', inplace_preprocess=self.inplace_preprocess,
                              cut_size=self.cut_size, random_shift=False, test=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True)

        self.data_loader['val'] = torch.utils.data.DataLoader(
            dataset=ds.Feeder(annotations_paths='val_annotations.pkl', inplace_preprocess=self.inplace_preprocess,
                              cut_size=self.cut_size, random_shift=False),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True)

    def get_model(self, input_shape=72):
        if self.model_type == 'lstm':
            return lstm(input_shape)
        elif self.model_type == 'gru':
            return gru(input_shape)
        else:
            raise Exception(f"{self.model_type} is not supported!")

    def process_x(self, x: np.array):
        if self.window_size:
            x = np.array(
                [np.concatenate((x[max(0, i - self.window_size): i + 1, :-1].mean(axis=0),
                                 time_step[-1:]))
                 for i, time_step in enumerate(x)]
            )

        x = x[-self.cut_size:]

        # logger.debug(f"{np.array(x).shape=}")
        x_swapped = [np.swapaxes(x, 1, 2).reshape(-1, 1, 2, x.shape[1])]
        x_pad = np.full((self.cut_size, x_swapped.shape[1], x_swapped.shape[2]), fill_value=-999)
        x_pad[-x.shape[0]:, :x.shape[1], :x.shape[2]] = x_swapped

        return x_pad

    def prepare_X_data(self, save=False):
        for selection in ['train', 'test', 'val']:
            anno = load_pickle(f"{selection}_annotations.pkl")

            for i, sample in enumerate(anno):
                path = sample['path']
                x = load_pickle(path)

                if i % 1000 == 0:
                    logger.debug(f"{selection=} | iter={i}")

                x_pad = self.process_x(x)

                if save:
                    save_pickle(path, x_pad)

    def test(self, cuda: bool = True, phase: str = 'test'):
        model = torch.load(self.checkpoint)
        self.model.load_state_dict(model['state_dict'])

        if cuda:
            self.model = self.model.cuda()

        dataloader = self.data_loader['test']
        self.model.eval()

        y_pred = list()
        y_test = list()
        paths = list()
        labels = list()

        for batch_idx, (data, label, path, or_labels) in enumerate(tqdm.tqdm(dataloader)):
            if cuda:
                data = data.float().cuda()
                label = label.float().cuda()
            # logger.debug(data.shape)

            with torch.no_grad():
                output = self.model(data)

                if isinstance(output, tuple):
                    output, l1 = output

            predict_label = torch.round(output.data)
            y_pred += [i[0] for i in predict_label.data.int().cpu().detach().tolist()]
            y_test += label.int().cpu().detach().tolist()
            labels += or_labels
            paths += path

        calc_metrics(y_test, y_pred, phase=phase, test=True)
        res = pd.DataFrame({'path': paths, 'y_pred': y_pred, 'y_test': y_test, 'label': labels})
        res.to_csv('/app/predictions.csv', index=False)

    def set_checkpoint(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint

    def grid_search(self, cuda: bool = True, n_iters: int = 20):
        model = torch.load(self.checkpoint)
        self.model.load_state_dict(model['state_dict'])

        if cuda:
            self.model = self.model.cuda()

        dataloader = self.data_loader['test']
        self.model.eval()

        y_pred = list()
        y_test = list()

        res = dict()

        best_score = float('-inf')
        best_threshold = 0.5

        for threshold in list(np.random.choice(np.arange(0.250, 0.999, 0.001), size=n_iters))+[0.5]:
            logger.debug(f"{threshold=}")
            for batch_idx, (data, label, _, _) in enumerate(tqdm.tqdm(dataloader)):
                if cuda:
                    data = data.float().cuda()
                    label = label.float().cuda()

                with torch.no_grad():
                    output = self.model(data.cuda())

                    if isinstance(output, tuple):
                        output, l1 = output

                #logger.debug(output.data)
                prob = output.data
                #logger.debug(pos_class_probs)
                predict_label = torch.where(prob > threshold, 1, 0)
                #logger.debug(predict_label)
                y_pred += [i[0] for i in predict_label.data.int().cpu().detach().tolist()]
                y_test += label.int().cpu().detach().tolist()

            score = calc_metrics(y_test, y_pred, phase='test', test=True)
            res[round(threshold, 4)] = round(score, 4)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        res = dict(sorted(res.items()))
        logger.debug(f"{res=}")
        logger.debug(f"{best_threshold=} | {best_score=}")


def calculate_rates(y_true, y_pred):
    cnf_matrix = confusion_matrix(y_true, y_pred)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)[0]
    FN = FN.astype(float)[0]
    TP = TP.astype(float)[0]
    TN = TN.astype(float)[0]

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)

    return f"{TPR=} | {TNR=} | {FPR=} | {FNR=}"


def calc_metrics(y_test, y_pred, phase, test=False) -> Tuple[float]:
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=np.unique(y_test))
    recall = recall_score(y_test, y_pred, labels=np.unique(y_test))
    fhalf_score = fbeta_score(y_test, y_pred, labels=np.unique(y_test), beta=0.5)
    f_score = fbeta_score(y_test, y_pred, labels=np.unique(y_test), beta=1.0)
    rates = calculate_rates(y_test, y_pred)
    MCC = matthews_corrcoef(y_test, y_pred)

    logger.debug(f"{phase=} | {phase}_{acc=} | {phase}_{precision=} | {phase}_{recall=}")
    logger.debug(f"{phase}_{f_score=} | {phase}_{fhalf_score=} | {phase}_{MCC=}")

    if test:
        logger.debug(rates)

    return fhalf_score


class lstm(nn.Module):
    def __init__(self, input_size, bidirectional=False):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True, dropout=0.5,
                            bidirectional=bidirectional)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)
        self.act = nn.ReLU(inplace=True)
        self.final = nn.Softmax()

    def forward(self, x):
        output, _status = self.lstm(x)
        output = output[:, -1, :]
        output = self.fc1(output)
        output = self.act(output)
        output = self.fc2(output)
        output = self.act(output)
        output = self.final(output)
        return output


class gru(nn.Module):
    def __init__(self, input_size, bidirectional=False):
        super(gru, self).__init__()
        self.gru_layer = nn.GRU(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True, dropout=0.5,
                                bidirectional=bidirectional)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)
        # self.fc3 = nn.Linear(in_features=16, out_features=1)
        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.ReLU(inplace=True)
        self.final = nn.Sigmoid()

    def forward(self, x):
        output, _status = self.gru_layer(x)
        output = output[:, -1, :]
        output = self.fc1(output)
        output = self.act(output)
        output = self.dropout(output)
        output = self.fc2(output)
        # output = self.act(output)
        # output = self.dropout(output)
        # output = self.fc3(output)
        output = self.final(output)
        return output


def main():
    loop = TrainingLoop(
        prepare_X=False, model_type='gru',
        cut_size=602, optimizer='Adam',
        warm_up_epoch=15, loss='bce',
        epochs=150
    )
    loop.train('output_directory')
    loop.set_checkpoint('output_directory/model_name/checkpoint_name.pt')
    loop.test()
    loop.grid_search(n_iters=150)


if __name__ == "__main__":
    main()
