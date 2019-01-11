import copy
from collections import deque
import itertools
import time
from pathlib import Path
import os

import multiprocessing as mp
import numpy as np
import pandas as pd
from PIL import Image
import PIL

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer

import albumentations as alb 

import math
import torchvision
from torchvision import transforms
import tqdm

import imgaug as ia
from imgaug import augmenters as iaa

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import matplotlib as mpl
mpl_params = {
    'figure.figsize': (10, 5),
    'figure.dpi': 300,
}
from matplotlib import pyplot as plt
mpl.rcParams.update(mpl_params)

import seaborn as sns
sns.set()

import visdom

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('../input/human-protein-atlas-image-classification/')
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR =  DATA_DIR / 'test'

train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df = pd.read_csv(DATA_DIR / 'sample_submission.csv')

HPA_DIR = Path('../input/HPAv18/')
hpa_df = pd.read_csv('../HPAv18RBGY_wodpl.csv')

#CHECKPOINT_PATH = Path('inceptionv3_512_nog_acc32x4_7norm-best_model-18.pth')
CHECKPOINT_PATH = Path('inceptionv3_512_nog_acc32x4_7norm-best_model-18.pth')

LOAD_CHECKPOINT = True 

TRAIN = False 
ADD_HPA = True 
NEGATIVE = False 

ONLY_VAL = False 
TRAIN_VAL = False 
VAL_TTA = 8

SUBMISSION_RUN = True 
TEST_TTA = 8

#ARCH = 'resnet18'
ARCH = 'inceptionv3'
N_EPOCHS = 25 
BATCH_SIZE = 32 * 8
LEARNING_RATE = 1e-3
SIGMOID_THRESHOLD = 0.5
VALIDATION_SIZE = .20
VISDOM_ENV_NAME = 'final_valtest'

NUM_WORKERS = mp.cpu_count()

vis = visdom.Visdom(env=VISDOM_ENV_NAME, server='http://3.17.85.107')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if ONLY_VAL:
    BATCH_SIZE *= 2

LABELS = {
    0: 'Nucleoplasm', 
    1: 'Nuclear membrane',   
    2: 'Nucleoli',   
    3: 'Nucleoli fibrillar center' ,  
    4: 'Nuclear speckles',
    5: 'Nuclear bodies',
    6: 'Endoplasmic reticulum',   
    7: 'Golgi apparatus',
    8: 'Peroxisomes',
    9: 'Endosomes',
    10: 'Lysosomes',
    11: 'Intermediate filaments',   
    12: 'Actin filaments',
    13: 'Focal adhesion sites',   
    14: 'Microtubules',
    15: 'Microtubule ends',   
    16: 'Cytokinetic bridge',   
    17: 'Mitotic spindle',
    18: 'Microtubule organizing center',  
    19: 'Centrosome',
    20: 'Lipid droplets',   
    21: 'Plasma membrane',   
    22: 'Cell junctions', 
    23: 'Mitochondria',
    24: 'Aggresome',
    25: 'Cytosol',
    26: 'Cytoplasmic bodies',   
    27: 'Rods & rings'
}

LABEL_NAMES = list(LABELS.values())

def augment_wrap(aug, image):
    return aug(image=image)['image']

class ProteinDataset(Dataset):
    def __init__(self, df, images_dir, transform=None, train=True, device='cpu', preproc=True):            
        self.df = df.copy()
        self.device = device
        self._dir = images_dir
        self.transform = transform
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([list(range(len(LABELS)))])
        self.colors = ['red', 'green', 'blue', 'yellow']
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def _load(self, path):
        with Image.open(path) as pil_im:
            return np.array(pil_im, np.uint8)
                                      
    def __getitem__(self, key):
        id_ = self.df.loc[key].Id
        is_additional_hpa = self.df.loc[key].get('hpa', False)
        choice = self.df.loc[key].get('choice', 'random')

        if is_additional_hpa:
            image_paths = [HPA_DIR / f'{id_}_{c}.png' for c in self.colors]
            rgb = np.array(Image.open(HPA_DIR / f'{id_}.png'), np.uint8)
        else:
            image_paths = [self._dir / f'{id_}_{c}.png' for c in self.colors]
            r, g, b, y = map(self._load, image_paths)
            rgb = np.stack([
                r,
                g,
                b,
            ], axis=2)
        if choice == 'shift':
            aug = alb.ShiftScaleRotate(shift_limit=0.01, interpolation=4, border_mode=0, scale_limit=(0, .1), rotate_limit=360, p=1)
            if self.train:
                rgb = augment_wrap(aug, rgb)
            choice = 0
        elif choice == 'rotate':
            aug = alb.Rotate(limit=360, interpolation=4, border_mode=0, p=1)
            if self.train:
                rgb = augment_wrap(aug, rgb)
            choice = 0
        X = self.transform(rgb)

        y = []
        if 'Target' in self.df:
            y = list(map(int, self.df.iloc[key].Target.split(' ')))
            y = self.mlb.transform([y]).squeeze()  # TODO: This is weird.
        
        if self.train:
            X = self.dihedral(X, choice)

        return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)

    def dihedral(self, x, choice='random'):
        if choice == 'random':
            choice = np.random.randint(8)
        if choice == 0:  # no rotation
            x = x
        elif choice == 1:  # 90
            x = np.rot90(x, k=1, axes=(1, 2)).copy()
        elif choice == 2:  # 180
            x = np.rot90(x, k=2, axes=(1, 2)).copy()
        elif choice == 3:  # 270
            x = np.rot90(x, k=3, axes=(1, 2)).copy()
        elif choice == 4:  # no rotation mirror
            x = np.rot90(x, k=0, axes=(1, 2)).copy()
            x = np.flip(x, axis=2).copy()
        elif choice == 5:  # 90 mirror
            x = np.rot90(x, k=1, axes=(1, 2)).copy()
            x = np.flip(x, axis=1).copy()
        elif choice == 6:  # 180 mirror
            x = np.rot90(x, k=2, axes=(1, 2)).copy()
            x = np.flip(x, axis=2).copy()
        elif choice == 7:  # 270 mirror
            x = np.rot90(x, k=3, axes=(1, 2)).copy()
            x = np.flip(x, axis=1).copy()
        return x


#aug_labels = [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 27]
#selection_df = hpa_df[hpa_df.Target.apply(lambda x: bool(set(aug_labels).intersection(list(map(int, x.split())))))]

train_split, val_split = train_test_split(train_df, test_size=VALIDATION_SIZE, random_state=0)

hpa_df['hpa'] = True
train_split['hpa'] = False

orig = val_split.copy()
if VAL_TTA:
    val_split['choice'] = 0
    for i in range(1, VAL_TTA):
        tmp = orig.copy()
        tmp['choice'] = i 
        val_split = val_split.append(tmp)
print(len(orig), ' -> ', len(val_split))

test_df_original = test_df.copy()
if TEST_TTA:
    test_df['choice'] = 0
    for i in range(1, TEST_TTA):
        tmp = test_df_original.copy()
        tmp['choice'] = i 
        test_df = test_df.append(tmp)
print(len(test_df_original), ' -> ', len(test_df))

if ADD_HPA:
    train_split = train_split.append(hpa_df)

def over_under_sampler(targets):
    n_samples = {
        0: 1, 25: 1,
        21: 1, 23: 1,
        2: 1, 4: 1, 5: 1, 7: 1,
        1: 1, 3: 1, 6: 1, 11: 1, 14: 1, 18: 1, 19: 1, 22: 1,
        12: 6, 13: 6, 16: 6,
        24: 16, 26: 16,
        8: 32, 9: 32, 10: 32, 15: 32, 20: 32, 27: 32, 17: 16,
    }
    multipliers = [n_samples[int(t)] for t in targets.split()]
    if targets == '0' or targets == '25 0' or targets == '25':
        return np.random.choice([0, 1], p=[.60, .40])
    elif '17' in targets:
        return 16
    elif 32 in multipliers:
        return 32
    else:
        return min(multipliers)

train_split['original_index'] = train_split.index
train_split = train_split.reset_index(drop=True)
val_split['original_index'] = val_split.index
val_split = val_split.reset_index(drop=True)
test_df['original_index'] = test_df.index
test_df = test_df.reset_index(drop=True)

train_split['oversample'] = train_split.Target.apply(over_under_sampler)
train_split = train_split.loc[train_split.index.repeat(train_split.oversample)]
del train_split['oversample']
train_split = train_split.reset_index(drop=True)

labels, counts = np.unique(list(map(int, itertools.chain(*train_split.Target.str.split()))), return_counts=True)
import numpy as np

name_label_dict = {} 

for i in range(28):
    name_label_dict[i] = (LABELS[i], counts[i])

print(name_label_dict)

n_labels = np.sum(counts)

def cls_wts(label_dict, mu=0.8):
    prob_dict, prob_dict_bal = {}, {}
    max_ent_wt = 1/28
    for i in range(28):
        prob_dict[i] = label_dict[i][1]/n_labels
        if prob_dict[i] > max_ent_wt:
            prob_dict_bal[i] = prob_dict[i]-mu*(prob_dict[i] - max_ent_wt)
        else:
            prob_dict_bal[i] = prob_dict[i]+mu*(max_ent_wt - prob_dict[i])            
    return prob_dict, prob_dict_bal

w1, w2 = cls_wts(name_label_dict)
print(w1)
print(w2)
w = [v for k, v in w2.items()]
weights2 = torch.tensor(w).cuda()
weights = 1. / torch.tensor(counts, dtype=torch.float, device=device)
print('weights orig')
print(weights2)
print('\n\nweights dampened')
print(weights)

def plot_labels(df, ax):
    labels, counts = np.unique(list(map(int, itertools.chain(*df.Target.str.split()))), return_counts=True)  
    pd.DataFrame(counts, labels).plot(kind='bar', ax=ax)
    ax.set_ylim([0, 30000])

#fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=200)
# plot_labels(train_df, ax1)
# plot_labels(train_split, ax2)
trans = [
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
]
#trans = trans if ARCH == 'resnet18' else trans[2:]
transform = transforms.Compose(trans[2:])
print(transform)

train_ds = ProteinDataset(
    train_split,
    images_dir=TRAIN_DIR,
    transform=transform,
    train=True,
    device=device,
    preproc=False,

)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

val_ds = ProteinDataset(
    val_split,
    images_dir=TRAIN_DIR,
    transform=transform,
    train=True,
    device=device,
    preproc=False,
)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS)

dataloaders = {'train': train_dl, 'val': val_dl}

if TRAIN_VAL:
    dataloaders['train'] = val_dl

test_ds = ProteinDataset(
    test_df,
    images_dir=TEST_DIR,
    transform=transform,
    train=True,
    device=device,
    preproc=False,
)
# TODO: Crash with pin_memory=True ...
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, pin_memory=False, num_workers=NUM_WORKERS)

class Inception3Adaptive(torchvision.models.Inception3):
    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # 1 x 1 x 4096
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 4096
        x = x.view(x.size(0), -1)
        # 4096
        x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x

class AdamW(Optimizer):
    """Implements AdamW algorithm.
    It has been proposed in `Fixing Weight Decay Regularization in Adam`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. Fixing Weight Decay Regularization in Adam:
    https://arxiv.org/abs/1711.05101
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # according to the paper, this penalty should come after the bias correction
                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'], p.data)

        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()

if ARCH == 'resnet18':
    model = torchvision.models.resnet18(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.fc = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.Linear(512, 28),
            )
    torch.nn.init.xavier_uniform_(model.fc[1].weight)
    
    model = torch.nn.DataParallel(model)
    beginning = [{'params': p, 'lr': 1e-5} for n, p in list(model.module.named_parameters())[:3]]
    one = [{'params': p, 'lr': 1e-4} for n, p in model.module.named_parameters() if n.startswith('layer1')]
    two = [{'params': p, 'lr': 1e-3} for n, p in model.module.named_parameters() if n.startswith('layer2')]
    three = [{'params': p, 'lr': 1e-3} for n, p in model.module.named_parameters() if n.startswith('layer3')]
    four = [{'params': p, 'lr': 1e-2} for n, p in model.module.named_parameters() if n.startswith('layer4')]
    fc = [{'params': p, 'lr': 1e-2} for n, p in model.module.named_parameters() if n.startswith('fc')]
    optim_params = beginning + one + two + three + four + fc

elif ARCH == 'inceptionv3':
    pretrained = torchvision.models.inception_v3(pretrained=True, transform_input=False)
    model = Inception3Adaptive(transform_input=False)
    model.load_state_dict(pretrained.state_dict())
    model.aux_logits = False
    model.fc = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 28),
        )
    torch.nn.init.xavier_uniform_(model.fc[1].weight)

    model = torch.nn.DataParallel(model)
    #first = [{'params': p, 'lr': 1e-4} for n, p in model.module.named_parameters() if n.startswith('Conv2d')]
    #middle = [{'params': p, 'lr': 1e-3} for n, p in model.module.named_parameters() if n[:7] in ['Mixed_5', 'Mixed_6', 'Mixed_7', 'AuxLogi']]
    #last = [{'params': p, 'lr': 1e-2} for n, p in model.module.named_parameters() if n.startswith('fc')]

    first = [{'params': p, 'lr': 1e-8} for n, p in model.module.named_parameters() if n.startswith('Conv2d')]
    middle = [{'params': p, 'lr': 1e-6} for n, p in model.module.named_parameters() if n[:7] in ['Mixed_5', 'Mixed_6', 'Mixed_7', 'AuxLogi']]
    last = [{'params': p, 'lr': 1e-4} for n, p in model.module.named_parameters() if n.startswith('fc')]
    optim_params = first + middle + last


sigmoid = nn.Sigmoid()

start_epoch = 0
if LOAD_CHECKPOINT:
    checkpoint = torch.load(CHECKPOINT_PATH)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    
for name, param in model.named_parameters():
    param.requires_grad = False

if TRAIN:    
    for name, param in model.named_parameters():
        param.requires_grad = True 

    criterion = nn.BCEWithLogitsLoss().cuda()
    #criterion = FocalLoss().cuda()
    
    optimizer = AdamW(params=optim_params, lr=LEARNING_RATE, weight_decay=1e-5)
    if LOAD_CHECKPOINT:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

model.to(device)

class RunningStats():
    def __init__(self):
        self.reset()

    def reset(self):
        self.latest = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 0

    def update(self, val, n=1):
        self.latest = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min
        
class TrainingStats():
    def __init__(self):
        self.train_loss = RunningStats()
        self.train_f1 = RunningStats()
        self.val_loss = RunningStats()
        self.val_f1 = RunningStats()
        
    def plot(self):
        pass  # TODO

# # Profile DataLoader / Estimate Train Time

# # train_ds.transform = resize_transform(128)
#load_stat = RunningStats()
#proc_stat = RunningStats()
#test_size, test_iter = 16, 100
#test_batch_size_dl = DataLoader(train_ds, batch_size=test_size, shuffle=True, pin_memory=True)
#t1 = time.time()
#model.train()
#optimizer.zero_grad()
#test_dl_iter = iter(test_batch_size_dl)
#for _ in tqdm.tqdm(range(test_iter)):
#    with torch.set_grad_enabled(True):
#        load_t1 = time.time()
#        X, y = next(test_dl_iter)
#        load_stat.update(time.time() - load_t1)
#        X = torch.as_tensor(X, dtype=torch.float32, device=device).cuda(non_blocking=True)
#        y = torch.as_tensor(y, dtype=torch.float32, device=device).cuda(non_blocking=True) if len(y) > 0 else y
#        proc_t1 = time.time()
#        y_ = model(X)
#        proc_stat.update(time.time() - proc_t1)
#        loss = criterion(y_, y)
#        loss.backward()
#        optimizer.step()
#batch_time = (time.time() - t1) / test_iter
#epoch_time = len(train_ds) / test_size * batch_time / 60
#train_time = epoch_time * N_EPOCHS / 60
#print(f'load time {load_stat.avg}')
#print(f'proc time {proc_stat.avg}')
#print(f'avg batch time: {batch_time:0.3f} s\nepoch est. {epoch_time:0.1f} m\ntrain est. {train_time:0.1f} h')
#quit()

def train(dataloaders, model, criterion, optimizer, sigmoid_thresh, n_epochs):
    start_ts = time.time()
    best_f1 = 0
    for epoch in range(n_epochs):
        stats = TrainingStats()
        total_epochs = start_epoch + epoch + 1
        print(f'Epoch {epoch + 1}/{n_epochs}')
        
        for phase in ['train', 'val']:
            hold_y = []
            hold_y_ = []
            if phase == 'train':
                if ONLY_VAL:
                    continue
                model.train()
            else:
                if TRAIN_VAL:
                    continue
                model.eval()

            optimizer.zero_grad()
            for i, (X, y) in enumerate(tqdm.tqdm(dataloaders[phase])):
                accumulation_steps = 16# if phase == 'train' else 8
                batch_weights = torch.Tensor([sum(weights[np.array(sample)]) for sample in test_ds.mlb.inverse_transform(y)])
                X = X.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True) if len(y) > 0 else y

                if phase == 'train' and NEGATIVE:
                    X2 = X.clone()
                    X2[:, 0, :, :] = 0
                    X2[:, 1, :, :] = 0
                    y2 = y.clone()
                    y2[:, :] = 0
                    X = torch.cat((X, X2), 0)
                    y = torch.cat((y, y2), 0)
                    batch_weights = torch.cat((batch_weights, batch_weights), 0)

                criterion.weights = batch_weights.cuda()

                with torch.set_grad_enabled(phase == 'train'):
                    y_ = model(X)
                    loss = criterion(y_, y) / accumulation_steps 

                if phase == 'train':
                    loss.backward()
                
                if (i + 1) % accumulation_steps == 0 and phase == 'train':
                        optimizer.step()
                        optimizer.zero_grad()

                if phase == 'train':
                    stats.train_loss.update(loss.item(), n=X.shape[0])
                    hold_y.append(y)
                    hold_y_.append(y_)
                else:
                    stats.val_loss.update(loss.item(), n=X.shape[0])
                    hold_y.append(y)
                    hold_y_.append(y_)

            f1, best_thresh = 0, 0
            top_5 = deque([], maxlen=5)
            hold_y = torch.cat(hold_y).cpu()
            hold_y_ = sigmoid(torch.cat(hold_y_)).cpu()
            print('hold_y_ before', hold_y_.shape)
            if VAL_TTA:
                hold_y = hold_y.reshape((VAL_TTA, -1, 28)).mean(dim=0)
                hold_y_ = hold_y_.reshape((VAL_TTA, -1, 28)).mean(dim=0)

            print('hold_y_ after', hold_y_.shape)

            for thresh in np.linspace(0, 1, 200):
                score = f1_score(hold_y, hold_y_ > thresh, average='macro')
                if score > f1:
                    top_5.append((score, thresh))
                    f1, best_thresh = score, thresh

            lb_thresh = torch.Tensor([
                0.362397820,0.043841336,0.075268817,0.059322034,0.075268817,
                0.075268817,0.043841336,0.075268817,0.010000000,0.010000000,
                0.010000000,0.043841336,0.043841336,0.014198783,0.043841336,
                0.010000000,0.028806584,0.014198783,0.028806584,0.059322034,
                0.010000000,0.126126126,0.028806584,0.075268817,0.010000000,
                0.222493880,0.028806584,0.010000000
            ])

            #for thresh in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            #    score = f1_score(hold_y, hold_y_ > lb_thresh * thresh, average='macro')
            #    if score > f1:
            #        top_5.append((score, 99))
            #        f1, best_thresh = score, thresh 

            for s, t in top_5:
                print(f'{s:.3f} - {t:.3f}', end=' ')

            print(f1, best_thresh)

            if phase == 'train':
                stats.train_f1.update(f1, n=X.shape[0])
                vis.line([stats.train_loss.avg], [total_epochs], update='append', opts={'title': 'Train Loss'}, win='Train Loss')
                vis.line([stats.train_f1.avg], [total_epochs], update='append', opts={'title': 'Train F1'}, win='Train F1')
                print(f'(train) loss: {stats.train_loss.avg} f1: {stats.train_f1.avg}')

                if TRAIN_VAL:
                    save_state = dict(
                        epoch=total_epochs,
                        state_dict=model.state_dict(),
                        f1=0,
                        loss=0,
                        optimizer=optimizer.state_dict(),
                    )
                    torch.save(save_state, f'{VISDOM_ENV_NAME}-best_model-{total_epochs}.pth')
            else:
                stats.val_f1.update(f1, n=X.shape[0])
                vis.line([stats.val_loss.avg], [total_epochs], update='append', opts={'title': 'Val Loss'}, win='Val Loss')
                vis.line([stats.val_f1.avg], [total_epochs], update='append', opts={'title': 'Val F1'}, win='Val F1')
                print(f'(val) loss: {stats.val_loss.avg} f1: {stats.val_f1.avg} thresh: {best_thresh}')
                
                if stats.val_f1.avg > best_f1 or epoch + 1 == n_epochs:
                    best_f1 = stats.val_f1.avg
                    save_state = dict(
                        epoch=total_epochs,
                        state_dict=model.state_dict(),
                        f1=stats.val_f1.avg,
                        loss=stats.val_loss.avg,
                        optimizer=optimizer.state_dict(),
                    )
                    torch.save(save_state, f'{VISDOM_ENV_NAME}-best_model-{total_epochs}.pth')

def evaluate(dl, model):
    model.eval()
    y_predictions = []

    for i, (X, _) in enumerate(tqdm.tqdm(dl)):
        X = X.cuda(non_blocking=True)

        with torch.set_grad_enabled(False):
            y_ = model(X)
            y_predictions.append(y_)

    y_predictions = sigmoid(torch.cat(y_predictions)).cpu()
    print(y_predictions.shape)

    if TEST_TTA:
        y_predictions = y_predictions.reshape((TEST_TTA, -1, 28)).mean(dim=0)
    y_predictions = y_predictions.numpy()
    print(y_predictions.shape)

    for t in [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5]:
        submission = test_df_original[['Id', 'Predicted']].copy()
        Predicted = []
        for i, prediction in enumerate(test_ds.mlb.inverse_transform(y_predictions > t)):
            if len(prediction) == 0:
                prediction = tuple([np.argmax(y_predictions[i])])
            all_labels = []
            for label in prediction:
                all_labels.append(str(label))
            Predicted.append(' '.join(all_labels))

        submission['Predicted'] = Predicted
        submission.to_csv(f'protein_classification{str(t * 10)}.csv', index=False)

    np.save(f'y_{CHECKPOINT_PATH}.npy', y_predictions)  # For offline work on setting thresholds

if TRAIN:
    train(dataloaders, model, criterion, optimizer, sigmoid_thresh=SIGMOID_THRESHOLD, n_epochs=N_EPOCHS)

if SUBMISSION_RUN:
    evaluate(test_dl, model)
