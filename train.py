import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from metrics import *
import pickle
import argparse
import torch.optim.lr_scheduler as lr_scheduler
from model import SocialImplicit
from trajectory_augmenter import TrajectoryAugmenter
from CFG import CFG

parser = argparse.ArgumentParser()

#Social-Loss specific parameters
parser.add_argument('--w_norm',
                    type=float,
                    default=0.0001,
                    help='Intra-distance loss weight')
parser.add_argument('--w_cos',
                    type=float,
                    default=0.0001,
                    help='Angle between nodes loss weight')
parser.add_argument('--w_trip',
                    type=float,
                    default=0.0001,
                    help='Triplet loss weight')

#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset',
                    default='hotel',
                    help='eth,hotel,univ,zara1,zara2,sdd')

#Training specifc parameters
parser.add_argument('--batch_size',
                    type=int,
                    default=128,
                    help='minibatch size')
parser.add_argument('--num_epochs',
                    type=int,
                    default=50,
                    help='number of epochs')
parser.add_argument('--clip_grad',
                    type=float,
                    default=None,
                    help='gadient clipping')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--lr_sh_rate',
                    type=int,
                    default=45,
                    help='number of steps to drop the lr')

parser.add_argument('--tag', default='tag', help='personal tag for the model ')
args = parser.parse_args()

print('*' * 30)
print("Training initiating....")
print(args)

#Social-Loss
loss_store = {"l2": 0, "gl2": 0, "gcos": 0, "trip": 0}

_l1_mean = nn.L1Loss()


def cdist_cosine_sim(a, b, eps=1e-08):
    a_norm = a / torch.clamp(a.norm(dim=1)[:, None], min=eps)
    b_norm = b / torch.clamp(b.norm(dim=1)[:, None], min=eps)
    return torch.acos(
        torch.clamp(torch.mm(a_norm, b_norm.transpose(0, 1)),
                    min=-1.0 + eps,
                    max=1.0 - eps))


def reset_loss_store():
    global loss_store
    loss_store = {"l2": 0, "gl2": 0, "gcos": 0, "trip": 0}


def implicit_likelihood_estimation_fast_with_trip_geo(V_pred, V_target):

    V_pred = V_pred.contiguous()

    diff = torch.abs(V_pred - V_target)

    diff_sum = torch.sum(diff, dim=(1, 2, 3))
    _, indices = torch.sort(diff_sum)
    min_indx = indices[0]
    V_pred_min = V_pred[min_indx]
    V_target = V_target.squeeze()

    error = _l1_mean(V_pred_min, V_target)
    trip_loss = _l1_mean(V_pred_min, V_pred[indices[1]]) - _l1_mean(
        V_pred_min, V_pred[indices[-1]])

    V_pred_min_ = V_pred_min.reshape(-1, 2)
    V_target_ = V_target.reshape(-1, 2)

    #Geometric distance length
    norm_loss = torch.abs(
        torch.cdist(V_pred_min_.unsqueeze(0), V_pred_min_.unsqueeze(0), p=2.0)
        - torch.cdist(V_target_.unsqueeze(0), V_target_.unsqueeze(0), p=2.0)
    ).mean()

    #Gemometric distance angle
    cos_loss = torch.abs(
        cdist_cosine_sim(V_pred_min_, V_pred_min_) -
        cdist_cosine_sim(V_target_, V_target_)).mean()

    loss_store["l2"] += error.item()
    loss_store["gl2"] += norm_loss.item()
    loss_store["gcos"] += cos_loss.item()
    loss_store["trip"] += trip_loss.item()

    return error + args.w_norm * norm_loss + args.w_trip * trip_loss + args.w_cos * cos_loss


def graph_loss(V_pred, V_target, V_obs):
    return implicit_likelihood_estimation_fast_with_trip_geo(V_pred, V_target)


#Data prep
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
data_set = './datasets/' + args.dataset + '/'

dset_train = TrajectoryDataset(data_set + 'train/',
                               obs_len=obs_seq_len,
                               pred_len=pred_seq_len,
                               skip=1,
                               norm_lap_matr=True)

loader_train = DataLoader(
    dset_train,
    batch_size=1,  #This is irrelative to the args batch size parameter
    shuffle=True,
    num_workers=0)

dset_val = TrajectoryDataset(data_set + 'val/',
                             obs_len=obs_seq_len,
                             pred_len=pred_seq_len,
                             skip=1,
                             norm_lap_matr=True)

loader_val = DataLoader(
    dset_val,
    batch_size=1,  #This is irrelative to the args batch size parameter
    shuffle=False,
    num_workers=1)

#Defining the model
is_eth = args.dataset == 'eth'
if is_eth:
    noise_weight = CFG["noise_weight_eth"]
else:
    noise_weight = CFG["noise_weight"]

model = SocialImplicit(spatial_input=CFG["spatial_input"],
                       spatial_output=CFG["spatial_output"],
                       temporal_input=CFG["temporal_input"],
                       temporal_output=CFG["temporal_output"],
                       bins=CFG["bins"],
                       noise_weight=noise_weight).cuda().double()

#Optimizer and Schedule
optimizer = optim.SGD(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=args.lr_sh_rate,
                                      gamma=0.1)

#Check pointing
checkpoint_dir = './checkpoint/' + args.tag + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)

print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)

#Training
metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}
trajaugmenter = TrajectoryAugmenter(data_loader=loader_train)


def train(epoch):
    global metrics, loader_train, loss_store
    model.train()

    total_loss = 0
    batch_loss = 0
    for cnt, batch in enumerate(loader_train):

        #Get data
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        V_obs, V_tr, obs_traj, pred_traj_gt = trajaugmenter.augment(
            V_obs, V_tr, obs_traj, pred_traj_gt)

        V_obs, V_tr, A_obs, obs_traj = V_obs.cuda().double(), V_tr.cuda(
        ).double(), A_obs.cuda().double(), obs_traj.cuda().double()

        optimizer.zero_grad()

        #Forward
        V_pred = model(V_obs.permute(0, 3, 1, 2), obs_traj)
        V_pred = V_pred.permute(0, 2, 3, 1)

        #Loss
        batch_loss += graph_loss(V_pred, V_tr, V_obs)
        total_loss += batch_loss.item()

        #Learn
        if cnt % args.batch_size == 0 and cnt != 0:
            batch_loss = batch_loss / args.batch_size
            batch_loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.clip_grad)
            optimizer.step()
            #Log
            print(args.tag, ' |TRAIN:', '\t Epoch:', epoch, '\t Batch loss:',
                  batch_loss.item())

            loss_store["l2"] /= args.batch_size
            loss_store["gl2"] /= args.batch_size
            loss_store["gcos"] /= args.batch_size
            print("Detailed train loss:", loss_store)
            reset_loss_store()

            #Reset
            batch_loss = 0
    metrics['train_loss'].append(total_loss / (cnt + 1))


iteration = 0


def vald():
    global metrics, loader_val, constant_metrics, iteration, loss_store
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for cnt, batch in enumerate(loader_val):

            #Get data
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch

            #Forward
            V_obs, V_tr, A_obs, obs_traj = V_obs.cuda().double(), V_tr.cuda(
            ).double(), A_obs.cuda().double(), obs_traj.cuda().double()

            V_pred = model(V_obs.permute(0, 3, 1, 2), obs_traj)
            V_pred = V_pred.permute(0, 2, 3, 1)

            #Loss
            total_loss += graph_loss(V_pred, V_tr, V_obs).item()

        print(args.tag, ' |VALD:', '\t Iteration:', iteration, '\t Loss:',
              total_loss / (cnt + 1))
        metrics['val_loss'].append(total_loss / (cnt + 1))
        loss_store["l2"] /= (cnt + 1)
        loss_store["gl2"] /= (cnt + 1)
        loss_store["gcos"] /= (cnt + 1)
        print("Detailed val loss:", loss_store)
        reset_loss_store()
        store_per = 0.05 * constant_metrics['min_val_loss']

        if (constant_metrics['min_val_loss'] -
                metrics['val_loss'][-1]) > store_per:
            # if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
            constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
            constant_metrics['min_val_epoch'] = iteration
            torch.save(model.state_dict(),
                       checkpoint_dir + 'val_best.pth')  # OK
    iteration += 1


print('Training started ...')
for epoch in range(args.num_epochs):
    train(epoch)
    vald()

    scheduler.step()

    print('*' * 30)
    print(args.tag, ' |Epoch:', args.tag, ":", epoch)
    for k, v in metrics.items():
        if len(v) > 0:
            print(k, v[-1])

    print(constant_metrics)
    print('*' * 30)
    for g in optimizer.param_groups:
        print("***------------->LR = ", g['lr'])
    with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp:
        pickle.dump(metrics, fp)

    with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)
