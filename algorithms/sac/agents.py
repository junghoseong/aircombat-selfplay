import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from ..sac import utils
from .networks import DDPGActor, SACActor, Critic, CriticParallelizedEnsemble
from .networks import AtariSharedEncoder as Encoder
from .networks import NoAug as ImgAug

class SACAgent:
    def __init__(self, obs_shape, action_shape, discount, device, lr, 
                 feature_dim, hidden_dim, critic_target_tau,
                 reward_scale_factor, use_tb, from_vision):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.discount = discount
        self.lr = lr
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.reward_scale_factor = reward_scale_factor
        self.use_tb = use_tb
        self.from_vision = from_vision
        # Changed log_std_bounds from [-10, 2] -> [-20, 2]
        self.log_std_bounds = [-20, 2]
        # Changed self.init_temperature to 1.0
        self.init_temperature = 1.0

        # models
        if self.from_vision:
            self.encoder = Encoder(obs_shape).to(device)
            model_repr_dim = self.encoder.repr_dim
        else:
            model_repr_dim = obs_shape[0]

        self.actor = SACActor(model_repr_dim, action_shape, feature_dim,
                        hidden_dim, self.log_std_bounds).to(device)
        
        self.critic = Critic(model_repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(model_repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # Changed target entropy from -dim(A) -> -dim(A)/2
        self.target_entropy = -action_shape[0] / 2.0
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        # optimizers
        if self.from_vision:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
            self.aug = ImgAug(pad=4)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()
    
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, training=True):
        self.training = training
        if self.from_vision:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, uniform_action=False, eval_mode=False):
        if self.from_vision:
            obs = utils.normalize_image(obs)
            obs = torch.from_numpy(obs).to(self.device)
            obs = self.encoder(obs)
        else:
            obs = torch.as_tensor(obs, device=self.device)

        dist = self.actor(obs)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
        
        if uniform_action:
            action.uniform_(-1.0, 1.0)

        # ## REFERENCE INPUT
        # action-spaces (manually coded)
        # action[0, 0, 0] = 20
        # action[0, 0, 1] = 0
        # action[0, 0, 2] = 20
        # action[0, 0, 3] = 30
        
        # action[0, 0, 4:] = 1
        # action[0, 0, -1] = 0
        # #print("action: ", action.cpu().numpy())
        
        
        return action.cpu().numpy()

    def update_critic(self, obs, action, reward, next_obs, step, not_done=None):
        metrics = dict()

        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_V -= self.alpha.detach() * log_prob
            # TODO: figure out whether we want the not_done at the end or not
            target_Q = self.reward_scale_factor * reward + \
                            (self.discount * target_V * not_done)


        Q1, Q2 = self.critic(obs,action)
        # scaled the loss by 0.5, might have some effect initially
        critic_loss = 0.5 * (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        if self.from_vision:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.from_vision:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q + (self.alpha.detach() * log_prob)
        actor_loss = actor_loss.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()

        return metrics

    def update_alpha(self, obs, step):
        metrics = dict()

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                    (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.use_tb:
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['alpha_loss'] = alpha_loss
            metrics['alpha_value'] = self.alpha

        return metrics

    def transition_tuple(self, replay_buffer, batch_size):
        batch_dict = replay_buffer.sample(batch_size)
        batch_list = []
        if self.from_vision:
            batch_dict['observations'] = utils.normalize_image(batch_dict['observations'])
            batch_dict['next_observations'] = utils.normalize_image(batch_dict['next_observations'])
        batch_list.append(batch_dict['observations'])
        batch_list.append(batch_dict['actions'])
        batch_list.append(batch_dict['rewards'])
        batch_list.append(batch_dict['next_observations'])
        batch_list.append(batch_dict['dones'])
        obs, action, reward, next_obs, terminal = utils.to_torch(batch_list, self.device)

        return (obs, action, reward, next_obs, terminal)

    def update(self, trans_tuple, step):
        metrics = dict()

        obs, action, reward, next_obs, terminal = trans_tuple

        not_done = ~terminal

        # augment
        if self.from_vision:
            obs = self.aug(obs)
            next_obs = self.aug(next_obs)
            # encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, next_obs, step, not_done))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update alpha
        metrics.update(self.update_alpha(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics


class EnsembleSACAgent(SACAgent):
    def __init__(self, obs_shape, action_shape, discount, device, lr, 
                 feature_dim, hidden_dim, critic_target_tau,
                 reward_scale_factor, use_tb, from_vision,
                 num_ensemble, ensemble_cfg):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.discount = discount
        self.lr = lr
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.reward_scale_factor = reward_scale_factor
        self.use_tb = use_tb
        self.from_vision = from_vision
        self.num_ensemble = num_ensemble
        if ensemble_cfg == 'simple':
            ensemble_cfg = SimpleNamespace(type='simple', utd_ratio=1, drop_rate=0.0, layer_norm=False, actor_update_ens_stat='min')
        assert ensemble_cfg.type in ['simple', 'redq', 'droq', 'edac']
        self.ensemble_cfg = ensemble_cfg
        # Changed log_std_bounds from [-10, 2] -> [-20, 2]
        self.log_std_bounds = [-20, 2]
        # Changed self.init_temperature to 1.0
        self.init_temperature = 1.0

        # models
        p = self.ensemble_cfg.drop_rate
        use_ln = self.ensemble_cfg.layer_norm
        if self.from_vision:
            self.encoder = Encoder(obs_shape, drop_rate=p, layer_norm=use_ln).to(device)
            model_repr_dim = self.encoder.repr_dim
        else:
            model_repr_dim = obs_shape[0]

        self.actor = SACActor(model_repr_dim, action_shape, feature_dim,
                        hidden_dim, self.log_std_bounds).to(device)

        # critic output shape: (ensemble_size, batch_size, 1)
        self.critic = CriticParallelizedEnsemble(model_repr_dim, action_shape, feature_dim,
                            hidden_dim, num_ensemble=self.num_ensemble, drop_rate=p, layer_norm=use_ln).to(device)
        self.critic_target = CriticParallelizedEnsemble(model_repr_dim, action_shape, feature_dim,
                                    hidden_dim, num_ensemble=self.num_ensemble, drop_rate=p, layer_norm=use_ln).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # Changed target entropy from -dim(A) -> -dim(A)/2
        self.target_entropy = -action_shape[0] / 2.0
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        # optimizers
        if self.from_vision:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
            self.aug = ImgAug(pad=4)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def update_critic(self, obs, action, reward, next_obs, step, not_done=None):
        metrics = dict()

        sampled_idxs = np.random.choice(self.num_ensemble, self.ensemble_cfg.subset_size, replace=False) \
            if self.ensemble_cfg.type == 'redq' else np.arange(self.num_ensemble)
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_Qs = self.critic_target(next_obs, next_action) # (ensemble_size, batch_size, 1)
            target_Qs = target_Qs[sampled_idxs] # (subset_size, batch_size, 1)
            target_V = torch.min(target_Qs, dim=0)[0] # (batch_size, 1)
            target_V -= self.alpha.detach() * log_prob
            # TODO: figure out whether we want the not_done at the end or not
            target_Q = self.reward_scale_factor * reward + \
                            (self.discount * target_V * not_done)
            
        Qs = self.critic(obs, action)
        # scaled the loss by 0.5, might have some effect initially
        td_loss = F.mse_loss(Qs, target_Q.expand(self.num_ensemble,-1,-1), reduction='none')
        td_loss = td_loss.mean(dim=(1,2)).sum()
        critic_loss = 0.5 * td_loss

        if self.ensemble_cfg.type == 'edac':
            if self.ensemble_cfg.eta > 0:
                obs_tile = obs.expand(self.num_ensemble,-1,-1)
                action_tile = action.expand(self.num_ensemble,-1,-1).requires_grad_(True)
                Qs_preds_tile = self.critic(obs_tile, action_tile)
                Qs_pred_grads, = torch.autograd.grad(Qs_preds_tile.sum(), action_tile, retain_graph=True, create_graph=True)
                Qs_pred_grads = Qs_pred_grads / (torch.norm(Qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
                Qs_pred_grads = Qs_pred_grads.transpose(0, 1) # (batch_size, ensemble_size, action_dim)

                Qs_pred_grads = torch.einsum('bik,bjk->bij', Qs_pred_grads, Qs_pred_grads) # cosine similarity
                masks = torch.eye(self.num_ensemble).to(self.device).expand(Qs_pred_grads.size(0),-1,-1)
                Qs_pred_grads = (1 - masks) * Qs_pred_grads
                grad_loss = torch.mean(torch.sum(Qs_pred_grads, dim=(1, 2))) / (self.num_ensemble - 1)

                critic_loss += 0.5 * self.ensemble_cfg.eta * grad_loss

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q/ens_mean'] = Qs.mean().item()
            metrics['critic_q/ens_std'] = Qs.std(dim=0).mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        if self.from_vision:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.from_vision:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        Qs = self.critic(obs, action)
        ens_stat = getattr(self.ensemble_cfg, 'actor_update_ens_stat', 'mean')
        if ens_stat == 'min':
            Q = Qs.min(dim=0)[0]
        elif ens_stat == 'mean':
            Q = Qs.mean(dim=0)
        elif ens_stat == 'max':
            Q = Qs.max(dim=0)[0]
        else:
            raise ValueError('Invalid ensemble stat: {}'.format(ens_stat))
        
        actor_loss = -Q + (self.alpha.detach() * log_prob)
        actor_loss = actor_loss.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()

        return metrics

    def transition_tuple(self, replay_buffer, batch_size):
        obses, actions, rewards, next_obses, terminals = [], [], [], [], []
        for _ in range(self.ensemble_cfg.utd_ratio):
            obs, action, reward, next_obs, terminal = super(EnsembleSACAgent, self).transition_tuple(replay_buffer, batch_size)
            obses.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_obses.append(next_obs)
            terminals.append(terminal)

        return (obses, actions, rewards, next_obses, terminals)

    def update(self, trans_tuple, step):
        metrics = dict()

        obses, actions, rewards, next_obses, terminals = trans_tuple

        # augment
        if self.from_vision:
            obses = [self.aug(obs) for obs in obses]
            next_obses = [self.aug(next_obs) for next_obs in next_obses]
            # encode
            obses = [self.encoder(obs) for obs in obses]
            with torch.no_grad():
                next_obses = [self.encoder(next_obs) for next_obs in next_obses]
        
        if self.use_tb:
            metrics['batch_reward'] = rewards[0].mean().item()

        # update critic
        for i in range(self.ensemble_cfg.utd_ratio):
            metrics.update(
                self.update_critic(obses[i], actions[i], rewards[i], next_obses[i], step, ~terminals[i]))

        # update actor
        metrics.update(self.update_actor(obses[0].detach(), step))

        # update alpha
        metrics.update(self.update_alpha(obses[0].detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics


class TD3Agent:
    def __init__(self, obs_shape, action_shape, discount, device, lr,
                 feature_dim, hidden_dim, target_tau, reward_scale_factor,
                 stddev_schedule, stddev_clip, use_tb, from_vision):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.discount = discount
        self.lr = lr
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.target_tau = target_tau
        self.reward_scale_factor = reward_scale_factor
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb
        self.from_vision = from_vision
        self.policy_delay = 2

        # models
        if self.from_vision:
            self.encoder = Encoder(obs_shape).to(device)
            model_repr_dim = self.encoder.repr_dim
        else:
            model_repr_dim = obs_shape[0]

        self.actor = DDPGActor(model_repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.actor_target = DDPGActor(model_repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(model_repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(model_repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        if self.from_vision:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
            self.aug = ImgAug(pad=4)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
 
        self.train()
        self.actor_target.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        if self.from_vision:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step=0, uniform_action=False, eval_mode=False):
        if self.from_vision:
            obs = utils.normalize_image(obs)
            obs = torch.from_numpy(obs).to(self.device)
            obs = self.encoder(obs)
        else:
            obs = torch.as_tensor(obs, device=self.device)

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
        
        if uniform_action:
            action.uniform_(-1.0, 1.0)
            
        return action.cpu().numpy()

    def update_critic(self, obs, action, reward, next_obs, step, not_done=None):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor_target(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = self.reward_scale_factor * reward + self.discount * target_V * not_done


        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        if self.from_vision:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
    
        if self.from_vision:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.mean
        
        Q1, _ = self.critic(obs, action)

        actor_loss = -Q1
        actor_loss = actor_loss.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()

        return metrics

    def transition_tuple(self, replay_buffer, batch_size):
        batch_dict = replay_buffer.sample(batch_size)
        batch_list = []
        if self.from_vision:
            batch_dict['observations'] = utils.normalize_image(batch_dict['observations'])
            batch_dict['next_observations'] = utils.normalize_image(batch_dict['next_observations'])
        batch_list.append(batch_dict['observations'])
        batch_list.append(batch_dict['actions'])
        batch_list.append(batch_dict['rewards'])
        batch_list.append(batch_dict['next_observations'])
        batch_list.append(batch_dict['dones'])
        obs, action, reward, next_obs, terminal = utils.to_torch(batch_list, self.device)

        return (obs, action, reward, next_obs, terminal)

    def update(self, trans_tuple, step):
        metrics = dict()

        obs, action, reward, next_obs, terminal = trans_tuple

        not_done = ~terminal

        # augment
        if self.from_vision:
            obs = self.aug(obs)
            next_obs = self.aug(next_obs)
            # encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, next_obs, step, not_done))

        if step%self.policy_delay == 0:
            # update actor
            metrics.update(self.update_actor(obs.detach(), step))

            # update critic target
            utils.soft_update_params(self.critic, self.critic_target,
                                    self.target_tau)

            # update actor target
            utils.soft_update_params(self.actor, self.actor_target,
                                    self.target_tau)

        return metrics