import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import float32
import numpy as np
import itertools

from .mlp import MLPBase
from .utils import check

class VAE(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim, latent_dim, #max_action = 1.0,
                 hidden_size=128):
        super(VAE, self).__init__()

        # embedding table
        # init_tensor = torch.rand(action_dim,
        #                          action_embedding_dim) * 2 - 1  # Don't initialize near the extremes.
        # self.embeddings = torch.nn.Parameter(init_tensor.type(float32), requires_grad=True)
        self.embeddings = torch.tensor(list(itertools.product([0, 1], repeat=discrete_action_dim)), dtype=torch.int)
        # print("self.embeddings", self.embeddings) 
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

class Action_representation(NeuralNet):
    def __init__(self,
                 state_dim,
                 discrete_action_dim,
                 continuous_action_dim,
                 #reduced_action_dim=2,
                 #reduce_parameter_action_dim=2,
                 latent_action_dim = 2,
                 embed_lr=1e-4,
                 ):
        super(Action_representation, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parameter_action_dim = continuous_action_dim
        #self.reduced_action_dim = reduced_action_dim
        self.state_dim = state_dim
        self.discrete_action_dim = discrete_action_dim
        # Action embeddings to project the predicted action into original dimensions
        # latent_dim=action_dim*2+parameter_action_dim*2
        self.latent_dim = latent_action_dim
        self.embed_lr = embed_lr
        self.vae = VAE(state_dim=self.state_dim, discrete_action_dim=self.action_dim,
                       continuous_action_dim=self.parameter_action_dim,
                       latent_dim=self.latent_dim, #max_action=1.0,
                       hidden_size=128).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-4)

    # def discrete_embedding(self,):
    #     emb = self.vae.embeddings

    #     return emb

    def unsupervised_loss(self, s1, a_d, a_c, s2, sup_batch_size, embed_lr): 

        a_d = a_d.to(self.device)

        s1 = s1.to(self.device)
        s2 = s2.to(self.device)
        a_c = a_c.to(self.device)

        vae_loss, recon_loss_d, recon_loss_c, KL_loss = self.train_step(s1, a_d, a_c, s2, sup_batch_size, embed_lr)
        return vae_loss, recon_loss_d, recon_loss_c, KL_loss

    def train_step(self, s1, a_d, a_c, s2, sup_batch_size, embed_lr=1e-4):
        state = s1
        action_d = a_d
        action_c = a_c
        next_state = s2
        vae_loss, recon_loss_s, recon_loss_c, KL_loss = self.loss(state, action_d, action_c, next_state,
                                                                  sup_batch_size)

        #self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=embed_lr)
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        return vae_loss.cpu().data.numpy(), recon_loss_s.cpu().data.numpy(), recon_loss_c.cpu().data.numpy(), KL_loss.cpu().data.numpy()

    def loss(self, state, action_d, action_c, next_state, sup_batch_size):
        
        recon_c, recon_s, mean, std = self.vae(state, action_d, action_c)

        recon_loss_s = F.mse_loss(recon_s, next_state, size_average=True) #dynamics predictive
        recon_loss_c = F.mse_loss(recon_c, action_c, size_average=True)

        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()

        # vae_loss = 0.25 * recon_loss_s + recon_loss_c + 0.5 * KL_loss
        # vae_loss = 0.25 * recon_loss_s + 2.0 * recon_loss_c + 0.5 * KL_loss  #best
        beta = 2.0
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
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            z = torch.FloatTensor(z.reshape(1, -1)).to(self.device)
            discrete_action = torch.FloatTensor(discrete_action.reshape(1, -1)).to(self.device)
            action_c, state = self.vae.decode(state, z, discrete_action)
        return action_c.cpu().data.numpy().flatten()   

    # def select_delta_state(self, state, z, action):
    #     with torch.no_grad():
    #         state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #         z = torch.FloatTensor(z.reshape(1, -1)).to(self.device)
    #         action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
    #         action_c, state = self.vae.decode(state, z, action)
    #     return state.cpu().data.numpy().flatten()
    def select_delta_state(self, state, z, action):
        with torch.no_grad():
            _, state = self.vae.decode(state, z, action)
        return state.cpu().data.numpy()

    # def get_embedding(self, action):
    #     # Get the corresponding target embedding
    #     action_emb = self.vae.embeddings[action]
    #     action_emb = torch.tanh(action_emb)
    #     return action_emb
    def pairwise_distances(x, y):
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

    def get_match_scores(self, action):
        # compute similarity probability based on L2 norm
        embeddings = self.vae.embeddings
        embeddings = torch.tanh(embeddings)
        action = action.to(self.device)
        # compute similarity probability based on L2 norm
        similarity = - self.pairwise_distances(action, embeddings)  # Negate euclidean to convert diff into similarity score
        return similarity

        # 获得最优动作，输出于embedding最相近的action 作为最优动作.

    def select_discrete_action(self, action):  #discrete action matching
        similarity = self.get_match_scores(action)
        val, pos = torch.max(similarity, dim=1)
        # print("pos",pos,len(pos))
        # if len(pos) == 1:
        #     return pos.cpu().item()  # data.numpy()[0]
        # else:
        #     # print("pos.cpu().item()", pos.cpu().numpy())
        #     return pos.cpu().numpy()
        return self.vae.embeddings[pos[0]].numpy()

    def save(self, filename, directory):
        torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))
        # torch.save(self.vae.embeddings, '%s/%s_embeddings.pth' % (directory, filename))

    def load(self, filename, directory):
        self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename), map_location=self.device))
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