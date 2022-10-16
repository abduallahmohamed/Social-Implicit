import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import glob
from utils import *
from metrics import *
from model import SocialImplicit
from CFG import CFG


def test(KSTEPS=20):

    global loader_test, model, ROBUSTNESS
    model.eval()
    ade_bigls = []
    fde_bigls = []
    step = 0
    for batch in loader_test:
        step += 1
        #Get data
        batch = [tensor.cuda().double() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch

        num_of_objs = obs_traj_rel.shape[1]

        V_tr = V_tr.squeeze()

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(
            V_obs.data.cpu().numpy().squeeze(), V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(
            V_tr.data.cpu().numpy().squeeze(), V_x[-1, :, :].copy())

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        V_predx = model(V_obs_tmp, obs_traj, KSTEPS=KSTEPS)

        for k in range(KSTEPS):
            V_pred = V_predx[k:k + 1, ...]

            V_pred = V_pred.permute(0, 2, 3, 1)

            V_pred = V_pred.squeeze()

            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(
                V_pred.data.cpu().numpy().squeeze(), V_x[-1, :, :].copy())
            #Sensitivity
            V_pred_rel_to_abs += ROBUSTNESS

            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_, fde_


for ROBUSTNESS in [0]:  #[-0.1, -0.01, 0, +0.01, +0.1]:
    print("*" * 30)
    print("*" * 30)
    print("ROBUSTNESS:", ROBUSTNESS)
    print("*" * 30)
    print("*" * 30)

    paths = [
        './checkpoint/social-implicit-eth',
        './checkpoint/social-implicit-hotel',
        './checkpoint/social-implicit-zara1',
        './checkpoint/social-implicit-zara2',
        './checkpoint/social-implicit-univ',
        './checkpoint/social-implicit-sdd',
    ]
    KSTEPS = 20

    EASY_RESULTS = []

    print("*" * 50)
    print('Number of samples:', KSTEPS)
    print("*" * 50)

    for feta in range(len(paths)):

        ade_ls = []
        fde_ls = []
        exp_ls = []
        path = paths[feta]
        exps = glob.glob(path)
        exps.sort()
        print('Models being tested are:', exps)

        for exp_path in exps:

            # try:

            print("*" * 50)
            print("Evaluating model:", exp_path)

            model_path = exp_path + '/val_best.pth'
            args_path = exp_path + '/args.pkl'
            with open(args_path, 'rb') as f:
                args = pickle.load(f)

            stats = exp_path + '/constant_metrics.pkl'
            with open(stats, 'rb') as f:
                cm = pickle.load(f)
            print("Stats:", cm)

            #Data prep
            obs_seq_len = args.obs_seq_len
            pred_seq_len = args.pred_seq_len
            data_set = './datasets/' + args.dataset + '/'

            dset_test = TrajectoryDataset(data_set + 'test/',
                                          obs_len=obs_seq_len,
                                          pred_len=pred_seq_len,
                                          skip=1,
                                          norm_lap_matr=True)

            loader_test = DataLoader(
                dset_test,
                batch_size=
                1,  #This is irrelative to the args batch size parameter
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
                                   noise_weight=noise_weight).cuda()
            model.load_state_dict(torch.load(model_path))
            model.cuda().double()
            model.eval()

            ade_ = 999999
            fde_ = 999999
            print("Testing ....")
            ad, fd = test(KSTEPS=KSTEPS)
            ade_ = min(ade_, ad)
            fde_ = min(fde_, fd)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            exp_ls.append(exp_path)
            print("ADE:", ade_, " FDE:", fde_)
        # except Exception as e:
        #     print(e)
        print("*" * 50)
        ade_ls = np.asarray(ade_ls)
        fde_ls = np.asarray(fde_ls)
        min_ade_indx = np.argmin(ade_ls)
        min_fde_indx = np.argmin(fde_ls)
        avg_ade_fde = (ade_ls + fde_ls) / 2.0
        min_avg_ade_fde = np.argmin(avg_ade_fde)

        EASY_RESULTS.append([
            exp_ls[min_avg_ade_fde],
            round(ade_ls[min_avg_ade_fde], 4),
            round(fde_ls[min_avg_ade_fde], 4)
        ])
    print(EASY_RESULTS)