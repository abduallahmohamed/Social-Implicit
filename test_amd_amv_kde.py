import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import *
from metrics import *
from model import SocialImplicit
from amd_amv_kde_metrics import calc_amd_amv, kde_lossf
from CFG import CFG


def test(KSTEPS=20):
    global loader_test, model, ROBUSTNESS
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step = 0
    mabs_loss = []
    kde_loss = []
    m_collect = []
    eig_collect = []
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

        b_samples = []
        for k in range(KSTEPS):
            V_pred = V_predx[k:k + 1, ...]

            V_pred = V_pred.permute(0, 2, 3, 1)

            V_pred = V_pred.squeeze()

            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(
                V_pred.data.cpu().numpy().squeeze(), V_x[-1, :, :].copy())

            #Sensitivity
            V_pred_rel_to_abs += ROBUSTNESS  #0.01 = 1 cm, 0.1 = 10 cm

            b_samples.append(V_pred_rel_to_abs[:, :, None, :].copy())

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

        abs_samples = np.concatenate(
            b_samples, axis=2)  #ab samples in (12,3,100,2) gt in (12,3,2)
        # print("Stacked Samples:", abs_samples.shape)

        m, nan_list, n_u, m_c, eig = calc_amd_amv(V_y_rel_to_abs.copy(),
                                                  abs_samples.copy())
        mabs_loss.append(m)  #m
        eig_collect.append(eig)
        _kde = kde_lossf(V_y_rel_to_abs.copy(), abs_samples.copy())
        kde_loss.append(_kde)
        m_collect.extend(m_c)  #m_c
        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)

    return ade_, fde_, sum(kde_loss) / len(kde_loss), sum(mabs_loss) / len(
        mabs_loss), sum(eig_collect) / len(eig_collect)


for ROBUSTNESS in [0]:  #, -0.1, -0.01, +0.01, +0.1]:
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
    KSTEPS = 1000
    EASY_RESULTS = []

    print("*" * 50)
    print('Number of samples:', KSTEPS)
    print("*" * 50)

    for feta in range(len(paths)):
        # try:
        ade_ls = []
        fde_ls = []
        exp_ls = []
        kde_ls = []
        amd_ls = []
        eig_ls = []
        path = paths[feta]
        exps = glob.glob(path)
        exps.sort()

        for exp_path in exps:

            model_path = exp_path + '/val_best.pth'
            args_path = exp_path + '/args.pkl'
            with open(args_path, 'rb') as f:
                args = pickle.load(f)

            stats = exp_path + '/constant_metrics.pkl'
            with open(stats, 'rb') as f:
                cm = pickle.load(f)

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
            model = model.cuda().double()
            model.eval()

            ade_ = 999999
            fde_ = 999999
            # print("Testing ....")
            ad, fd, kd, md, eg = test(KSTEPS=KSTEPS)
            ade_ = min(ade_, ad)
            fde_ = min(fde_, fd)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            kde_ls.append(kd)
            amd_ls.append(md)
            exp_ls.append(exp_path)
            eig_ls.append(eg)
            print("amd,kde,amv:", md, kd, eg)
        # except:
        # pass
        print("*" * 50)

        ade_ls = np.asarray(ade_ls)
        fde_ls = np.asarray(fde_ls)
        kde_ls = np.asarray(kde_ls)
        amd_ls = np.asarray(amd_ls)
        eig_ls = np.asarray(eig_ls)

        min_ade_indx = np.argmin(ade_ls)
        min_fde_indx = np.argmin(fde_ls)
        avg_ade_fde = (ade_ls + fde_ls) / 2.0
        min_avg_ade_fde = np.argmin(avg_ade_fde)

        min_kde_indx = np.argmin(kde_ls)
        min_amd_indx = np.argmin(amd_ls)
        min_eig_indx = np.argmin(eig_ls)
        avg_kde_mde = (kde_ls + amd_ls) / 2.0
        min_avg_kde_ade = np.argmin(avg_kde_mde)

        avg_eig_mde = (eig_ls + amd_ls) / 2.0
        min_avg_eig_ade = np.argmin(avg_eig_mde)

        # print("Min ADE:", np.min(ade_ls), " at:", exp_ls[min_ade_indx])
        # print("Min FDE:", np.min(fde_ls), " at:", exp_ls[min_fde_indx])
        # print("Min ADE/FDE:", np.min(avg_ade_fde), " at:",
        #       exp_ls[min_avg_ade_fde], " with ADE/FDE:",
        #       ade_ls[min_avg_ade_fde], "|", fde_ls[min_avg_ade_fde])

        # print("Min KDE:", np.min(kde_ls), " at:", exp_ls[min_kde_indx])
        # print("Min AMD:", np.min(amd_ls), " at:", exp_ls[min_amd_indx])
        # print("Min EIG:", np.min(eig_ls), " at:", exp_ls[min_eig_indx])

        # print("Min KDE/AMD:", np.min(avg_kde_mde), " at:",
        #       exp_ls[min_avg_kde_ade], " with EIG/AMD/KDE:",
        #       eig_ls[min_avg_kde_ade], "|", amd_ls[min_avg_kde_ade], "|",
        #       kde_ls[min_avg_kde_ade])

        # print("Min EIG/AMD:", np.min(avg_eig_mde), " at:",
        #       exp_ls[min_avg_eig_ade], " with EIG/AMD/KDE:",
        #       eig_ls[min_avg_eig_ade], "|", amd_ls[min_avg_eig_ade], "|",
        #       kde_ls[min_avg_eig_ade])

        EASY_RESULTS.append([
            exp_ls[min_avg_eig_ade],
            round(amd_ls[min_avg_eig_ade], 4),
            round(kde_ls[min_avg_eig_ade], 4), eig_ls[min_avg_eig_ade]
        ])

    # except Exception as e:
    #     print(e, "Error in:", feta)
    for kkkk in EASY_RESULTS:
        print(kkkk)