import random
import numpy as np
import pandas as pd
#from .mdn import MDN
from mbcd.utils.dataset import Dataset, normalize, denormalize
from mbcd.utils.logger import Logger
from itertools import combinations
import pickle

from .models.constructor import construct_model


class MBCD:

    def __init__(self, state_dim, action_dim, sac=None, n_hidden_units=200, num_layers=4, memory_capacity=100000, cusum_threshold=300, max_std=0.5, num_stds=2, run_id=None):
        self.run_id = run_id

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.sac = sac

        self.n_hidden_units = n_hidden_units
        self.num_layers = num_layers

        self.memory_capacity = memory_capacity
        self.memory = Dataset(state_dim, action_dim, memory_capacity)

        self.threshold = cusum_threshold
        self.max_std = max_std
        self.num_stds = num_stds
        self.min_steps = 5000
        self.changed = False
        self.step = 0

        self.num_models = 1
        self.current_model = 0
        self.steps_per_context = {0: 0}
        self.models = {0: self._build_model(0)}
        self.log_prob = {0: 0.0}
        self.var_mean = {0: 0.0}
        self.S = {0: 0.0, -1: 0.0} # -1 is statistic for new model
        self.mean, self.variance = {}, {}

        if self.sac is not None:
            self.sac.save('weights/'+run_id+'init_pi')

        self.logger = Logger()
        self.test_mode = False

    @property
    def counter(self):
        return self.steps_per_context[self.current_model]

    def _build_model(self, id):
        if id != 0:
            return construct_model(name='BNN'+self.run_id+str(id), obs_dim=self.state_dim, act_dim=self.action_dim, num_networks=5, num_elites=2,
                                    session=self.models[0].sess)
        return construct_model(name='BNN'+self.run_id+str(id), obs_dim=self.state_dim, act_dim=self.action_dim, num_networks=5, num_elites=2)

    def train(self):
        X, Y = self.memory.to_train_batch()
        self.models[self.current_model].train(X, Y, batch_size=256, holdout_ratio=0.1)

    def get_logprob2(self, x, means, variances):
        '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
        '''
        k = x.shape[-1]

        mean = np.mean(means, axis=0)
        variance = (np.mean(means**2 + variances, axis=0) - mean**2) +1e-6

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k*np.log(2*np.pi) + np.log(variance).sum(-1) + (np.power(x-mean, 2)/variance).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(axis=0)

        ## [ batch_size ]
        log_prob = np.log(prob + 1e-8)  # Avoid log of zero

        #var_mean = np.var(means, axis=0).max(axis=-1)
        var_mean = np.linalg.norm(np.std(means, axis=0), axis=-1)

        """ if self.models[0].scaler.fitted:
            mu, sigma = self.models[0].scaler.cached_mu[0,:-self.action_dim], self.models[0].scaler.cached_sigma[0,:-self.action_dim]
            std_means = (means[:,:,:-1] - mu) / sigma
            var_mean = np.var(std_means, axis=0).max(axis=-1)
        else:
            var_mean = [1] """

        #maxes = np.max(np.linalg.norm(variances, axis=-1), axis=0)
        """
        disc = -1
        for i, j in combinations(range(means.shape[0]), 2):
            disc = max(disc, np.linalg.norm(means[i][0] - means[j][0])) """

        return log_prob, var_mean[0], mean, variance

    def update_metrics(self, obs, action, reward, next_obs, done):
        obs = obs[None]
        action = action[None]
        inputs = np.concatenate((obs, action), axis=-1)
        true_output = np.concatenate(([reward], next_obs))[None]

        preds = []
        if self.changed:
            self.S = {m: 0.0 for m in range(self.num_models)}
            self.S[-1] = 0.0

        for i in range(self.num_models):
            ensemble_model_means, ensemble_model_vars = self.models[i].predict(inputs, factored=True)
            preds.append(ensemble_model_means.copy())
            ensemble_model_means[:,:,1:] += obs
            self.log_prob[i], self.var_mean[i], self.mean[i], self.variance[i] = self.get_logprob2(true_output, ensemble_model_means, ensemble_model_vars)

        #print(self.mean[self.current_model][0][0], reward)
        for i in (m for m in range(self.num_models) if m != self.current_model):
            if self.var_mean[self.current_model] < self.max_std and self.counter > self.min_steps:
                log_ratio = self.log_prob[i] - self.log_prob[self.current_model]  # log(a/b) = log(a) - log(b)
                self.S[i] = max(0, self.S[i] + log_ratio)

        new_model_log_pdf = -1/2 * ((self.state_dim+1)*np.log(2*np.pi) + \
                        np.log(self.variance[self.current_model]).sum(-1) + \
                        (np.power(true_output-(true_output+self.num_stds*np.sqrt(self.variance[self.current_model])), 2) / \
                        self.variance[self.current_model]).sum(-1))
        new_model_log_pdf = np.log(np.exp(new_model_log_pdf).sum(0) + 1e-8)

        if self.var_mean[self.current_model] < self.max_std and self.counter > self.min_steps:
            log_ratio =  new_model_log_pdf - self.log_prob[self.current_model]
            self.S[-1] = max(0, self.S[-1] + log_ratio)

        changed = False
        maxm = max(self.S.values())
        if maxm > self.threshold:
            changed = True
            self.memory.remove_last_n(n=100) # Remove last experiences, as they may be from different context

            if maxm == self.S[-1]:  # New Model
                newm = self.new_model()
                self.set_model(newm, load_params_from_init_model=True)
            else:
                newm = max(self.S, key=lambda key: self.S[key])
                self.set_model(newm)

        self.changed = changed
        self.step += 1
        self.steps_per_context[self.current_model] += 1

        predreward, preddelta = preds[0][0][0][0], preds[0][0][0][1:]
        d = {}
        for f in range(self.state_dim):
            d['f'+str(f)] = obs[0][f]
            d['preddeltaf'+str(f)] = preddelta[f]
            d['deltaf'+str(f)] = next_obs[f] - obs[0][f]
        d['predreward'] = predreward
        d['done'] = done
        d['reward'] = reward
        d['null'] = -new_model_log_pdf
        d.update({'neglogp'+str(m): -self.log_prob[m] for m in range(self.num_models)})
        d.update({'s'+str(m): self.S[m] for m in range(self.num_models)})
        d.update({'var'+str(m): self.var_mean[m] for m in range(self.num_models)})

        for i in range(self.num_models, 4):
            d['neglogp'+str(i)] = np.nan
            d['s'+str(i)] = np.nan
            d['var'+str(i)] = np.nan

        d['s_uniform'] = self.S[-1]
        self.logger.log(d)

        if self.step % 50000 == 0:
            if self.test_mode:
                    self.logger.save('results/' + self.run_id + 'H{}-14'.format(self.threshold))
            else:
                self.logger.save('results/' + self.run_id)

        return changed
    
    def update_metrics_mbpo(self, obs, action, reward, next_obs, done):
        obs = obs[None]
        action = action[None]
        inputs = np.concatenate((obs, action), axis=-1)
        true_output = np.concatenate(([reward], next_obs))[None]

        preds = []

        for i in range(self.num_models):
            ensemble_model_means, ensemble_model_vars = self.models[i].predict(inputs, factored=True)
            preds.append(ensemble_model_means.copy())
            ensemble_model_means[:,:,1:] += obs
            self.log_prob[i], self.var_mean[i], self.mean[i], self.variance[i] = self.get_logprob2(true_output, ensemble_model_means, ensemble_model_vars)

        self.step += 1
        self.steps_per_context[self.current_model] += 1

        predreward, preddelta = preds[0][0][0][0], preds[0][0][0][1:]
        d = {}
        for f in range(self.state_dim):
            d['f'+str(f)] = obs[0][f]
            d['preddeltaf'+str(f)] = preddelta[f]
            d['deltaf'+str(f)] = next_obs[f] - obs[0][f]
        d['predreward'] = predreward
        d['done'] = done
        d['reward'] = reward
        d.update({'neglogp'+str(m): -self.log_prob[m] for m in range(self.num_models)})
        d.update({'var'+str(m): self.var_mean[m] for m in range(self.num_models)})

        self.logger.log(d)

        if self.step % 5000 == 0:
            if self.test_mode:
                    self.logger.save('results/' + self.run_id + 'test')
            else:
                self.logger.save('results/' + self.run_id)

        return False

    def update_metrics_oracle(self, obs, action, reward, next_obs, done, task):
        self.step += 1

        changed = False
        if task == 'wind' or task == 'normal':
            context = 0
        if task == 'joint-malfunction':
            context = 1
        if task == 'velocity':
            context = 2
        if context != self.current_model:
            self.set_model(context)
            changed = True

        d = {}
        d['reward'] = reward
        self.logger.log(d)

        if self.step % 100 == 0:
            if self.test_mode:
                    self.logger.save('results/' + self.run_id + 'oracle9')

        return changed

    def predict(self, sa, model=None):
        model = self.current_model if model is None else model
        mean, var = self.models[model].predict(sa)
        return mean

    def new_model(self):
        self.steps_per_context[self.num_models] = 0
        self.models[self.num_models] = self._build_model(self.num_models)
        self.log_prob[self.num_models] = 0.0
        self.S[self.num_models] = 0.0
        self.var_mean[self.num_models] = 0.0
        #self.maxes[self.num_models] = 0.0
        #self.disc[self.num_models] = 0.0
        self.num_models += 1
        return self.num_models - 1

    def set_model(self, model_id, load_params_from_init_model=False):
        if not self.test_mode:
            self.save_current()

        self.current_model = model_id
        # new model
        if load_params_from_init_model:
            #self.sac.load_parameters('weights/'+self.run_id+'init_pi')
            self.memory = Dataset(self.state_dim, self.action_dim, self.memory_capacity)
        # load existent model
        else:
            if self.sac is not None:
                self.sac.load_parameters('weights/'+self.run_id+'pi'+str(self.current_model))
            self.load_dataset(self.current_model)

    def add_experience(self, obs, actions, rewards, next_obs, dones):
        self.memory.push(obs, actions, rewards, next_obs, dones)

    def save_current(self):
        if self.sac is not None:
            self.sac.save('weights/'+self.run_id+'pi'+str(self.current_model))
        self.save_dataset(self.current_model)
        self.save_model(self.current_model)
    
    def save_model(self, i):
        self.models[i].save_weights()

    def save_dataset(self, i):
        with open('weights/'+self.run_id+'data'+str(i), 'wb') as f:
            pickle.dump(self.memory, f)

    def load_dataset(self, i):
        with open('weights/'+self.run_id+'data'+str(i), 'rb') as f:
            self.memory = pickle.load(f)

    def save_models(self):
        for i in range(self.num_models):
            self.models[i].save_weights()

    def load(self, num_models, load_policy=True):
        for i in range(num_models):
            if i != 0:
                self.new_model()
            self.models[i].load_weights()
            self.steps_per_context[i] = self.min_steps + 1  # So it will not ignore context detection
        self.load_dataset(self.current_model)
        if load_policy:
            self.sac.load_parameters('weights/'+self.run_id+'pi'+str(self.current_model))



""" def exps_to_train_batch(obs, actions, rewards, dones, bootstrap=False):
    X = []; Y = []
    if bootstrap:
        inds = np.random.choice(len(obs)-1, len(obs)-1)
    else:
        inds = np.arange(len(obs)-1)

    for i in inds:
        if not dones[i]:
            X.append(np.hstack((obs[i], actions[i])))
            Y.append(np.hstack((obs[i+1] - obs[i], [rewards[i], dones[i+1]])))
    X, Y = np.stack(X), np.stack(Y)
    return X, Y

def exps_to_train_seqs(obs, actions, dones, seq_len, skip_ahead):
    X, Y = [], []
    obs_and_actions = []
    for timestep in range(len(obs)):
        obs_and_actions.append(np.concatenate((obs[timestep], actions[timestep])))
    for j in range(0, len(obs) - seq_len, skip_ahead):
        if not np.any(dones[j:j+seq_len]):
            X.append(obs_and_actions[j:j+seq_len])   # The N prev obs. and actions
            Y.append(obs[j+1:j+seq_len+1])           # The next observations
    X, Y = np.array(X), np.array(Y)
    return X, Y """
