import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
import numpy as np

from .mlp import MLPBase
from .utils import check
from ..utils.buffer import SharedHybridReplayBuffer

class Discriminator(nn.Module):
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        super(Discriminator, self).__init__()

        # train config
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=self.device)

        # PPO config
        self.ppo_epoch = args.ppo_epoch
        self.hidden_size = args.hidden_size
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.recurrent_hidden_size = args.recurrent_hidden_size

        # Network config
        self.num_agents = num_agents
        if isinstance(act_space, spaces.Box):
            self.input_dim = 2 * act_space.shape[0] + self.recurrent_hidden_size
            self.input_dim_wo_act = act_space.shape[0] + self.recurrent_hidden_size
        else: 
            self.input_dim = 2*sum(len(s.nvec) for s in act_space.spaces) + self.recurrent_hidden_size
            self.input_dim_wo_act = sum(len(s.nvec) for s in act_space.spaces) + self.recurrent_hidden_size

        self.output_dim = obs_space.shape[0]

        self.pred = check(predict_net(self.input_dim, self.output_dim, args)).to(**self.tpdv)
        self.pred_wo = check(predict_net(self.input_dim_wo_act, self.output_dim, args)).to(**self.tpdv)

        self.q_optimizer = optim.Adam(list(self.pred.parameters()) + list(self.pred_wo.parameters()), lr=args.lr)
        self.q_loss_coef = 0.01

    def train(self, buffer):
        """
        Main training loop where we iterate over the data generator
        to get mini-batches of length L (chunk length) for RNN training.
        """
        train_info = {}
        train_info['disc_loss'] = 0

        # Repeat training for a certain number of PPO epochs
        for _ in range(self.ppo_epoch):
            # Retrieve the RNN-based mini-batch data generator
            data_generator = buffer.random_batch_generator(self.num_mini_batch, self.data_chunk_length)

            # For each chunk of the sequence data
            for sample in data_generator:
                if isinstance(buffer,SharedHybridReplayBuffer):
                    obs_batch, next_obs_batch, share_obs_batch, _, _, _, actions_batch, _, _, masks_batch, active_masks_batch, rnn_states_actor_batch = sample
                else:
                    obs_batch, next_obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, rnn_states_actor_batch = sample
                

                # Update qϕ (influence estimator) and pη (prediction network)
                disc_loss = self.update_parameters(obs_batch, next_obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, rnn_states_actor_batch)
                train_info['disc_loss'] += disc_loss.item()

        # Calculate the average disc_loss over all updates
        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info


    def update_parameters(self, obs_batch, next_obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, rnn_states_actor_batch):

        # get shape
        max_batch_size, rollout_threads, agent_num, _ = obs_batch.shape

        # shape : (max_batch_size, rollout_threads, agent_num, ...)
        obs_batch = check(obs_batch).to(**self.tpdv)  
        next_obs_batch = check(next_obs_batch).to(**self.tpdv)
        share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        rnn_states_actor_batch = check(rnn_states_actor_batch).to(**self.tpdv)

        loss_influence = 0.0
        loss_prediction = 0.0

        for thread in range(rollout_threads):
            ### agent 1 ###
            q_input = torch.cat([
                rnn_states_actor_batch[:,thread,0,:],      
                actions_batch[:,thread,0,:],
                actions_batch[:,thread,1,:]
            ], dim=-1)

            q_pred = self.pred(q_input) 
            loss_influence += F.mse_loss(q_pred, next_obs_batch[:,thread,0,:])

            q_input_wo= torch.cat([
                rnn_states_actor_batch[:,thread,0,:],      
                actions_batch[:,thread,0,:]
            ], dim=-1)

            q_pred_wo = self.pred_wo(q_input_wo) 
            loss_influence += F.mse_loss(q_pred_wo, next_obs_batch[:,thread,0,:])

            ### agent 2 ###
            q_input = torch.cat([
                rnn_states_actor_batch[:,thread,0,:],      
                actions_batch[:,thread,1,:],
                actions_batch[:,thread,0,:]
            ], dim=-1)

            q_pred = self.pred(q_input)  
            loss_influence += F.mse_loss(q_pred, next_obs_batch[:,thread,1,:])

            q_input_wo= torch.cat([
                rnn_states_actor_batch[:,thread,0,:],      
                actions_batch[:,thread,1,:]
            ], dim=-1)

            q_pred_wo = self.pred_wo(q_input_wo)
            loss_influence += F.mse_loss(q_pred_wo, next_obs_batch[:,thread,1,:])

        # Optionally, you can average by (L-1)*N
        loss_influence = loss_influence / (rollout_threads)

        # Compute loss
        total_loss = self.q_loss_coef * loss_influence 

        self.q_optimizer.zero_grad()
        total_loss.backward()
        self.q_optimizer.step()

        return total_loss


    def compute_intrinsic_reward(self, obs, next_obs, actions, rewards, dones, rnn_states_actor):

        # get shape
        rollout_threads, agent_num, _ = obs.shape

        # initialize intrinsic reward
        reward_int = torch.zeros_like(torch.tensor(rewards, dtype=torch.float32))  # (threads, agents, 1)

        # shape : (max_batch_size, rollout_threads, agent_num, ...)
        obs = check(obs).to(**self.tpdv)  
        next_obs = check(next_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        dones = check(dones).to(**self.tpdv)
        rnn_states_actors = check(rnn_states_actor).squeeze(dim=2).to(**self.tpdv)

        ### agent 1 ###

        q_input = torch.cat([
            rnn_states_actors[:,0,:],      
            actions[:,0,:],
            actions[:,1,:]
        ], dim=-1)

        log_prob = self.pred.get_log_pi(q_input, next_obs[:,0,:])

        q_input_wo= torch.cat([
            rnn_states_actors[:,0,:],      
            actions[:,0,:]
        ], dim=-1)

        log_prob_wo = self.pred_wo.get_log_pi(q_input_wo, next_obs[:,0,:])

        reward_int[:,1,:] = log_prob - log_prob_wo

        ### agent 2 ###
        q_input = torch.cat([
            rnn_states_actors[:,0,:],      
            actions[:,1,:],
            actions[:,0,:]
        ], dim=-1)

        log_prob = self.pred.get_log_pi(q_input, next_obs[:,1,:])


        q_input_wo= torch.cat([
            rnn_states_actors[:,0,:],      
            actions[:,1,:]
        ], dim=-1)

        log_prob_wo = self.pred_wo.get_log_pi(q_input_wo, next_obs[:,1,:])

        reward_int[:,0,:] = log_prob - log_prob_wo

        return reward_int.detach().cpu().numpy()


    def compute_MI(self, obs, next_obs, actions, dones, rnn_states_actor):

        # get shape
        agent_num, _ = obs.shape

        # initialize intrinsic reward
        MI_array = torch.zeros((agent_num, 1), dtype=torch.float32)  # (agents, 1)

        # shape : (agent_num, ...)
        obs = check(obs).to(**self.tpdv)  
        next_obs = check(next_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        dones = check(dones).to(**self.tpdv)
        rnn_states_actors = check(rnn_states_actor).squeeze().to(**self.tpdv)

        ### agent 1 ###

        q_input = torch.cat([
            rnn_states_actors,      
            actions[0,:],
            actions[1,:]
        ], dim=-1)

        log_prob = self.pred.get_log_pi(q_input, next_obs[0,:])

        q_input_wo= torch.cat([
            rnn_states_actors,      
            actions[0,:]
        ], dim=-1)

        log_prob_wo = self.pred_wo.get_log_pi(q_input_wo, next_obs[0,:])

        MI_array[1,:] = log_prob - log_prob_wo

        ### agent 2 ###
        q_input = torch.cat([
            rnn_states_actors,      
            actions[1,:],
            actions[0,:]
        ], dim=-1)

        log_prob = self.pred.get_log_pi(q_input, next_obs[1,:])


        q_input_wo= torch.cat([
            rnn_states_actors,      
            actions[1,:]
        ], dim=-1)

        log_prob_wo = self.pred_wo.get_log_pi(q_input_wo, next_obs[1,:])

        MI_array[0,:] = log_prob - log_prob_wo

        return MI_array.squeeze().detach().cpu().numpy()


class predict_net(nn.Module):

    def __init__(self, input_dim, output_dim, args):
        super(predict_net, self).__init__()

        self.output_dim = output_dim

        self.layers = nn.ModuleList()
        self.hidden_size = list(map(int, args.hidden_size.split()))
        self.layers.append(nn.Linear(input_dim, self.hidden_size[0]*2))

        for i in range(1, len(self.hidden_size)):
            self.layers.append(nn.Linear(self.hidden_size[i - 1]*2, self.hidden_size[i]*2))

        self.last_fc = nn.Linear(self.hidden_size[-1]*2, self.output_dim)

    def forward(self, x):

        for layer in self.layers:
            x = F.relu(layer(x))

        x = self.last_fc(x)
        return x

    def get_log_pi(self, own_variable, other_variable):
        predict_variable = self.forward(own_variable)
        log_prob = -1 * F.mse_loss(predict_variable, other_variable)
        log_prob = torch.sum(log_prob, -1, keepdim=False)

        return log_prob