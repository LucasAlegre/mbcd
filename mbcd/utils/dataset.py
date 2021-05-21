import numpy as np


class Dataset:

    def __init__(self, obs_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr, self.size, = 0, 0

        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rews_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((max_size, 1), dtype=np.float32)
        
        self.input_mean, self.output_mean = None, None
        self.input_std, self.output_std = None, None
        
    def push(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = action
        self.rews_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def remove_last_n(self, n):
        self.ptr -= n

    def sample(self, batch_size, replace=True):
        inds = np.random.choice(self.size, batch_size, replace=replace)
        return self.obs_buf[inds], self.acts_buf[inds], self.rews_buf[inds], self.next_obs_buf[inds], self.done_buf[inds]

    def to_train_batch(self, normalization=False):
        inds = np.arange(self.size)

        X = np.hstack((self.obs_buf[inds], self.acts_buf[inds]))
        Y = np.hstack((self.rews_buf[inds], self.next_obs_buf[inds] - self.obs_buf[inds]))

        """ if normalization:
            self.input_mean = np.mean(X, axis=0)
            self.input_std = np.std(X, axis=0)
            self.output_mean = np.mean(Y, axis=0)
            self.output_std = np.std(Y, axis=0)
            X = normalize(X, self.input_mean, self.input_std)
            Y = normalize(Y, self.output_mean, self.output_std) """

        return X, Y

    def __len__(self):
        return self.size

def normalize(data, mean, std):
    return (data - mean) / (std + 1e-10)

def denormalize(data, mean, std):
    return data * (std + 1e-10) + mean