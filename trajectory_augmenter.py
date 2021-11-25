import torch
import torch.distributions as tdist
import numpy as np


class TrajectoryAugmenter():
    def __init__(self, total_time=20, split_time=8, data_loader=None):
        self.total_time = total_time
        self.split_time = split_time
        self.choice_dist = tdist.uniform.Uniform(0, 1)
        self.data_loader = data_loader
        self.bin = 1 / 7

    def abs_to_rel_split(self, v, full_traj, _split=8):
        v = torch.cat((torch.zeros(v.shape[0], v.shape[1], v.shape[2],
                                   1), v[:, :, :, 1:] - v[:, :, :, :-1]),
                      dim=-1).permute(0, 3, 1, 2)
        return v[:, :_split, ...], v[:, _split:, ...], full_traj[
            ..., :self.split_time], full_traj[...,
                                              self.split_time:]  #v_obs, v_tr

    def augment(self, V_obs, V_tr, obs_traj, pred_traj_gt):
        decision = self.choice_dist.sample().item()
        if (0 <= decision < self.bin):
            return V_obs, V_tr, obs_traj, pred_traj_gt
        elif self.bin <= decision < self.bin * 2:
            return self._aug_jitter(obs_traj, pred_traj_gt)
        elif self.bin * 2 <= decision < self.bin * 3:
            return self._aug_flip_mirror(obs_traj, pred_traj_gt)
        elif self.bin * 3 <= decision < self.bin * 4:
            return self._aug_flip_reverse(obs_traj, pred_traj_gt)
        elif self.bin * 4 <= decision < self.bin * 5:
            return self._aug_nodes(V_obs, V_tr, obs_traj, pred_traj_gt)
        elif self.bin * 5 <= decision < self.bin * 6:
            return self._aug_rot(obs_traj, pred_traj_gt)
        elif self.bin * 6 <= decision <= self.bin * 7:
            return self._aug_speed(obs_traj, pred_traj_gt)

    def _aug_jitter(self, obs_traj, pred_traj_gt, jit_per=0.1):

        u = tdist.uniform.Uniform(torch.Tensor([-jit_per, -jit_per]),
                                  torch.Tensor([jit_per, jit_per]))
        full_traj = torch.cat((obs_traj, pred_traj_gt), dim=-1)
        return self.abs_to_rel_split(
            full_traj +
            u.sample(sample_shape=(self.total_time, )).T.to(obs_traj.device),
            full_traj, self.split_time)

    def _aug_flip_mirror(self, obs_traj, pred_traj_gt):

        full_traj = torch.cat((obs_traj, pred_traj_gt), dim=-1)
        full_traj = torch.flip(full_traj, [2, 3])
        return self.abs_to_rel_split(full_traj, full_traj, self.split_time)

    def _aug_flip_reverse(self, obs_traj, pred_traj_gt):

        full_traj = torch.cat((obs_traj, pred_traj_gt), dim=-1)
        full_traj = torch.flip(full_traj, [3])
        return self.abs_to_rel_split(full_traj, full_traj, self.split_time)

    def _aug_nodes(self,
                   V_obs,
                   V_tr,
                   obs_traj,
                   pred_traj_gt,
                   int_low=1,
                   int_high=4):
        V_obs_lst = [V_obs]
        V_tr_lst = [V_tr]
        obs_traj_lst = [obs_traj]
        pred_traj_gt_lst = [pred_traj_gt]
        for i in range(
                torch.randint(low=int_low, high=int_high, size=(1, )).item()):
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs_, A_obs, V_tr_, A_tr = next(
                iter(self.data_loader))
            V_obs_lst.append(V_obs_)
            V_tr_lst.append(V_tr_)
            obs_traj_lst.append(obs_traj)
            pred_traj_gt_lst.append(pred_traj_gt)

        V_obs = torch.cat(V_obs_lst, dim=2)
        V_tr = torch.cat(V_tr_lst, dim=2)
        obs_traj = torch.cat(obs_traj_lst, dim=1)
        pred_traj_gt = torch.cat(pred_traj_gt_lst, dim=1)
        return V_obs, V_tr, obs_traj, pred_traj_gt

    def _aug_rot(self, obs_traj, pred_traj_gt):
        degrees = [
            0.2617993877991494, 0.5235987755982988, 0.7853981633974483,
            1.0471975511965976, 1.3089969389957472, 1.5707963267948966,
            1.8325957145940461, 2.0943951023931953, 2.356194490192345,
            2.6179938779914944, 2.8797932657906435, 3.141592653589793,
            3.4033920413889427, 3.6651914291880923, 3.9269908169872414,
            4.1887902047863905, 4.4505895925855405, 4.71238898038469,
            4.974188368183839, 5.235987755982989, 5.497787143782138,
            5.759586531581287, 6.021385919380437
        ]
        rot_degree = degrees[torch.randint(0, 23, (1, )).item()]
        rot_matrix = torch.Tensor([[np.cos(rot_degree),
                                    np.sin(rot_degree)],
                                   [-np.sin(rot_degree),
                                    np.cos(rot_degree)]]).double()
        full_traj = torch.cat((obs_traj, pred_traj_gt), dim=-1)
        full_traj = torch.matmul(
            rot_matrix,
            full_traj.transpose(2,
                                3).unsqueeze(-1)).squeeze(-1).transpose(2, 3)
        return self.abs_to_rel_split(full_traj, full_traj, self.split_time)

    def _aug_speed(self, obs_traj, pred_traj_gt, inc_distance=1):
        inc = tdist.uniform.Uniform(torch.Tensor([-inc_distance]),
                                    torch.Tensor([inc_distance
                                                  ])).sample().item()
        full_traj = torch.cat((obs_traj, pred_traj_gt), dim=-1) + torch.arange(
            0, inc, inc / 20).to(obs_traj.device)
        return self.abs_to_rel_split(full_traj, full_traj, self.split_time)
