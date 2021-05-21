import numpy as np
import tensorflow as tf
import pdb

def alive_bonus(self, z, pitch):
    return +1 if z > 0.8 and abs(pitch) < 1.0 else -1

def termination_fn_hopper_roboschool(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    z = next_obs[:, 0] + 1.25
    pitch = next_obs[:, 7]
    not_done = np.isfinite(next_obs).all(axis=-1) * (z > 0.8) * (pitch < 1.0)
    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  np.isfinite(next_obs).all(axis=-1) \
                * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                * (height > .7) \
                * (np.abs(angle) < .2)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_false(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done


class FakeEnv:

    def __init__(self, model, env_id):
        self.model = model
        self.termination_func = termination_fn_false
        """ if env_id == 'Hopper-v2':
            self.termination_func = termination_fn_hopper
        elif env_id == 'HalfCheetah-v2' or env_id == 'Reacher-v2' or env_id == 'ReacherNS-v2' or env_id == 'PusherNS-v2' or env_id == 'Pusher-v2' or env_id == 'CartPole-v0' or env_id == 'Maze-v0':
            self.termination_func = termination_fn_false
        elif env_id == 'HopperGravityBulletEnv-v0' or env_id.startswith('SunblazeHopper'):
            self.termination_func = termination_fn_hopper_roboschool
        else:
            raise NotImplementedError
         """
    def _get_logprob(self, x, means, variances):
        '''
            x : [ batch_size, obs_dim + 1 ]
            means : [ num_models, batch_size, obs_dim + 1 ]
            vars : [ num_models, batch_size, obs_dim + 1 ]
        '''
        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        #stds = np.std(means,0).mean(-1)
        #var_mean = np.var(means, axis=0, ddof=1).mean(axis=-1)
        maxes = np.max(np.linalg.norm(variances, axis=-1), axis=0)

        return log_prob, maxes

    def step(self, obs, act, deterministic=False):
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        #### choose one model from ensemble
        num_models, batch_size, _ = ensemble_model_means.shape
        model_inds = self.model.random_inds(batch_size)
        batch_inds = np.arange(0, batch_size)
        samples = ensemble_samples[model_inds, batch_inds]
        model_means = ensemble_model_means[model_inds, batch_inds]
        model_stds = ensemble_model_stds[model_inds, batch_inds]
        ####

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.termination_func(obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:,:1], terminals, model_means[:,1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:,:1], np.zeros((batch_size,1)), model_stds[:,1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info

    ## for debugging computation graph
    def step_ph(self, obs_ph, act_ph, deterministic=False):
        assert len(obs_ph.shape) == len(act_ph.shape)

        inputs = tf.concat([obs_ph, act_ph], axis=1)
        # inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.create_prediction_tensors(inputs, factored=True)
        # ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means = tf.concat([ensemble_model_means[:,:,0:1], ensemble_model_means[:,:,1:] + obs_ph[None]], axis=-1)
        # ensemble_model_means[:,:,1:] += obs_ph
        ensemble_model_stds = tf.sqrt(ensemble_model_vars)
        # ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            # ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
            ensemble_samples = ensemble_model_means + tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

        samples = ensemble_samples[0]

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.config.termination_ph_fn(obs_ph, act_ph, next_obs)
        info = {}

        return next_obs, rewards, terminals, info

    def close(self):
        pass



