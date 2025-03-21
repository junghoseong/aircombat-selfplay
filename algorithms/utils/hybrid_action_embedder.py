import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import float32
import numpy as np
import itertools

from .mlp import MLPBase
from .utils import check, get_shape_from_space
from ..utils.buffer import SharedHybridReplayBuffer
from .distributions import BetaShootBernoulli, Categorical

class VAE(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim, latent_dim, #max_action = 1.0,
                 hidden_size=128):
        super(VAE, self).__init__()

        # embedding table
        # init_tensor = torch.rand(action_dim,
        #                          action_embedding_dim) * 2 - 1  # Don't initialize near the extremes.
        # self.embeddings = torch.nn.Parameter(init_tensor.type(float32), requires_grad=True)
        # print("self.embeddings", self.embeddings) 
        #print("state_dim + discrete_action_dim",state_dim + discrete_action_dim)
        self.e0_0 = nn.Linear(state_dim + discrete_action_dim, hidden_size)
        self.e0_1 = nn.Linear(continuous_action_dim, hidden_size)

        self.e1 = nn.Linear(hidden_size, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        self.d0_0 = nn.Linear(state_dim + discrete_action_dim, hidden_size)
        self.d0_1 = nn.Linear(latent_dim, hidden_size)
        self.d1 = nn.Linear(hidden_size, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)

        self.continuous_action_output = nn.Linear(hidden_size, continuous_action_dim)

        self.d3 = nn.Linear(hidden_size, hidden_size)

        self.delta_state_output = nn.Linear(hidden_size, state_dim)

        #self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, discrete_action, continuous_action):

        z_0 = F.relu(self.e0_0(torch.cat([state, discrete_action], 1)))
        z_1 = F.relu(self.e0_1(continuous_action))   # parameter, state, action.
        z = z_0 * z_1 

        z = F.relu(self.e1(z))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)

        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        u, s = self.decode(state, z, discrete_action)

        return u, s, mean, std

    def decode(self, state, z=None, discrete_action=None, clip=None, raw=True):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(device)
            if clip is not None:
                z = z.clamp(-clip, clip)
        v_0 = F.relu(self.d0_0(torch.cat([state, discrete_action], 1)))
        v_1 = F.relu(self.d0_1(z))
        v = v_0 * v_1
        v = F.relu(self.d1(v))
        v = F.relu(self.d2(v))

        continuous_action_decoded = self.continuous_action_output(v)

        v = F.relu(self.d3(v))
        state_shift= self.delta_state_output(v)

        if raw: return continuous_action_decoded, state_shift
        #return self.max_action * torch.tanh(continuous_action_decoded), torch.tanh(state_shift)

class Action_representation(nn.Module):
    def __init__(self,
                 args,
                 obs_space,
                 rnn_actor_space,
                 discrete_action_space,
                 continuous_action_space,
                 #reduced_action_dim=2,
                 #reduce_parameter_action_dim=2,
                 continuous_embedding_space,
                 embed_lr=1e-4,
                 device=torch.device("cpu")
                 ):
        super(Action_representation, self).__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device = self.device)
        self.parameter_action_dim = get_shape_from_space(continuous_action_space)[0]
        #self.reduced_action_dim = reduced_action_dim
        self.state_dim = get_shape_from_space(obs_space)[0] + rnn_actor_space.shape[1]
        #print("self.state_dim",self.state_dim)
        self.discrete_action_dim = get_shape_from_space(discrete_action_space)[0]
        #print("self.discrete_action_dim",self.discrete_action_dim)
        # Action embeddings to project the predicted action into original dimensions
        # latent_dim=action_dim*2+parameter_action_dim*2
        product = list(itertools.product([0, 1], repeat=self.discrete_action_dim))
        self.embeddings = torch.tensor(product, dtype=torch.int32).to(**self.tpdv)
        self.latent_dim = get_shape_from_space(continuous_embedding_space)[0]
        self.embed_lr = embed_lr
        self.vae = VAE(state_dim=self.state_dim, discrete_action_dim=self.discrete_action_dim,
                       continuous_action_dim=self.parameter_action_dim,
                       latent_dim=self.latent_dim, #max_action=1.0,
                       hidden_size=128).to(**self.tpdv)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-4)
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length

    # def discrete_embedding(self,):
    #     emb = self.vae.embeddings

    #     return emb

    def train(self, buffer):
        train_info = {}
        train_info['vae_total_loss'] = 0
        train_info['vae_dynamics_predictive_loss'] = 0 #####Need to be eliminated
        train_info['vae_action_reconstruct_loss'] = 0
        train_info['vae_KL_loss'] = 0

        for _ in range(self.ppo_epoch):
            
            data_generator = buffer.random_batch_generator(self.num_mini_batch, self.data_chunk_length)

            # For each chunk of the sequence data
            for sample in data_generator:
                if isinstance(buffer,SharedHybridReplayBuffer):
                    obs_batch, next_obs_batch, share_obs_batch, next_share_obs_batch, discrete_actions_batch, continuous_actions_batch,\
                      actions_batch, _, _, masks_batch, active_masks_batch, rnn_states_actor_batch, next_rnn_states_actor_batch= sample
                else:
                    obs_batch, next_obs_batch, discrete_actions_batch, continuous_actions_batch,\
                      actions_batch, _, _, masks_batch, active_masks_batch, rnn_states_actor_batch, next_rnn_states_actor_batch= sample
                state_pre = np.concatenate((obs_batch, rnn_states_actor_batch), axis = -1)
                state_after = np.concatenate((next_obs_batch,next_rnn_states_actor_batch), axis = -1)
                del obs_batch, next_obs_batch, rnn_states_actor_batch, next_rnn_states_actor_batch

                #print("state_pre",state_pre.shape)
                #print("state_after",state_after.shape)

                _,_,_,state_pre_shape = state_pre.shape
                _,_,_,discrete_actions_shape = discrete_actions_batch.shape
                _,_,_,continuous_actions_shape = continuous_actions_batch.shape
                _,_,_,state_after_shape = state_after.shape

                state_pre = state_pre.reshape(-1,state_pre_shape)
                discrete_actions_batch = discrete_actions_batch.reshape(-1,discrete_actions_shape)
                continuous_actions_batch = continuous_actions_batch.reshape(-1,continuous_actions_shape)
                state_after = state_after.reshape(-1,state_after_shape)

                state_pre = torch.tensor(state_pre).to(**self.tpdv)
                discrete_actions_batch = torch.tensor(discrete_actions_batch).to(**self.tpdv)
                continuous_actions_batch = torch.tensor(continuous_actions_batch).to(**self.tpdv)
                state_after = torch.tensor(state_after).to(**self.tpdv)

                vae_loss, recon_loss_d, recon_loss_c, KL_loss = self.train_step(state_pre,discrete_actions_batch,continuous_actions_batch,state_after,1e-4)  

                train_info['vae_total_loss'] += vae_loss
                train_info['vae_dynamics_predictive_loss'] += recon_loss_d
                train_info['vae_action_reconstruct_loss'] += recon_loss_c
                train_info['vae_KL_loss'] += KL_loss

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        del data_generator

        return train_info




    def unsupervised_loss(self, s1, a_d, a_c, s2, sup_batch_size, embed_lr): 

        a_d = torch.tensor(a_d).to(**self.tpdv)

        s1 = torch.tensor(s1).to(**self.tpdv)
        s2 = torch.tensor(s2).to(**self.tpdv)
        a_c = torch.tensor(a_c).to(**self.tpdv)

        vae_loss, recon_loss_d, recon_loss_c, KL_loss = self.train_step(s1, a_d, a_c, s2, sup_batch_size, embed_lr)
        return vae_loss, recon_loss_d, recon_loss_c, KL_loss

    def train_step(self, state, action_d, action_c, next_state, embed_lr=1e-4):
        #vae_loss, recon_loss_s, recon_loss_c, KL_loss = self.loss(state, action_d, action_c, next_state,0)
        #self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=embed_lr)
        recon_c, recon_s, mean, std = self.vae(state, action_d, action_c)

        recon_loss_s = F.mse_loss(recon_s, next_state, size_average=True) #dynamics predictive
        recon_loss_c = F.mse_loss(recon_c, action_c, size_average=True)

        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()

        beta = 0.25
        vae_loss = beta * recon_loss_s + recon_loss_c + 0.5 * KL_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        return vae_loss.cpu().data.numpy(), recon_loss_s.cpu().data.numpy(), recon_loss_c.cpu().data.numpy(), KL_loss.cpu().data.numpy()

    def loss(self, state, action_d, action_c, next_state, sup_batch_size):
        # print("input in training vae")
        # print("state",state.shape)
        # print("action_d",action_d.shape)
        # print("action_c",action_c.shape)
        # print("next_state",next_state.shape)
        recon_c, recon_s, mean, std = self.vae(state, action_d, action_c)

        recon_loss_s = F.mse_loss(recon_s, next_state, size_average=True) #dynamics predictive
        recon_loss_c = F.mse_loss(recon_c, action_c, size_average=True)

        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()

        # vae_loss = 0.25 * recon_loss_s + recon_loss_c + 0.5 * KL_loss
        # vae_loss = 0.25 * recon_loss_s + 2.0 * recon_loss_c + 0.5 * KL_loss  #best
        beta = 0.25
        vae_loss = beta * recon_loss_s + 2.0 * recon_loss_c + 0.5 * KL_loss  ##beta should be adjusted here,
        # print("vae loss",vae_loss)
        # return vae_loss, 0.25 * recon_loss_s, recon_loss_c, 0.5 * KL_loss
        # return vae_loss, 0.25 * recon_loss_s, 2.0 * recon_loss_c, 0.5 * KL_loss #best
        return vae_loss, recon_loss_s, 2.0 * recon_loss_c, 0.5 * KL_loss

    
    def select_continuous_action(self, state, z, discrete_action):
        """
            select continuous action from state, latent vector & discrete actions
        """
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(**self.tpdv)
            if z == []:
                raise ValueError("z should not be empty")
            z = torch.FloatTensor(z.reshape(1, -1)).to(**self.tpdv)
            discrete_action = torch.FloatTensor(discrete_action).reshape(1, -1).to(**self.tpdv)

            action_c, state = self.vae.decode(state, z, discrete_action)
        return action_c.cpu().data.numpy().flatten()   


    def select_delta_state(self, state, z, action):
        with torch.no_grad():
            _, state = self.vae.decode(state, z, action)
        return state.cpu().data.numpy()

    # def get_embedding(self, action):
    #     # Get the corresponding target embedding
    #     action_emb = self.vae.embeddings[action]
    #     action_emb = torch.tanh(action_emb)
    #     return action_emb
    def pairwise_distances(self, x, y):
        '''
            Input: x is a Nxd matrix
                y is a Mxd matirx
            Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

            Advantage: Less memory requirement O(M*d + N*d + M*N) instead of O(N*M*d)
            Computationally more expensive? Maybe, Not sure.
            adapted from: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        '''


        x_norm = (x ** 2).sum(1).view(-1, 1)   
        y_norm = (y ** 2).sum(1).view(1, -1)

        y_t = torch.transpose(y, 0, 1)  
        # a^2 + b^2 - 2ab
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)    
        
        return dist


    def select_discrete_action(self, action):  #discrete action matching ####TODO : put this into ShootBernoulli. Then why use hybrid action?
        #embeddings = self.embeddings.to(**self.tpdv)
        action = torch.tensor(action)
        if action.dim() == 1:
            action = action.view(-1,self.discrete_action_dim)
        action = action.to(**self.tpdv)
        # compute similarity probability based on L2 norm
        similarity = - self.pairwise_distances(action, self.embeddings)

        val, pos = torch.max(similarity, dim=1)
        #print("pos",pos,len(pos))
        # if len(pos) == 1:
        #     return pos.cpu().item()  # data.numpy()[0]
        # else:
        #     # print("pos.cpu().item()", pos.cpu().numpy())
        #     return pos.cpu().numpy()
        
        return self.embeddings[pos].cpu().numpy()

    def save(self, filename, directory):
        torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))
        # torch.save(self.vae.embeddings, '%s/%s_embeddings.pth' % (directory, filename))

    def load(self, filename):
        #self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename), map_location=self.device))
        self.vae.load_state_dict(torch.load(filename))
        # self.vae.embeddings = torch.load('%s/%s_embeddings.pth' % (directory, filename), map_location=self.device)

    def get_c_rate(self, s1, a_d, a_c, s2, batch_size=100, range_rate=5):  #boundary??
        a_d = self.get_embedding(a_d).to(self.device)
        s1 = s1.to(self.device)
        s2 = s2.to(self.device)
        a_c = a_c.to(self.device)
        _, recon_s, mean, std = self.vae(s1, a_d, a_c) #
        # print("recon_s",recon_s.shape)
        z = mean + std * torch.randn_like(std)    #mean + standard deviated latent vector
        z = z.cpu().data.numpy()
        c_rate = self.z_range(z, batch_size, range_rate)
        # print("s2",s2.shape)

        recon_s_loss = F.mse_loss(recon_s, s2, size_average=True)

        # recon_s = abs(np.mean(recon_s.cpu().data.numpy()))
        return c_rate, recon_s_loss.detach().cpu().numpy()