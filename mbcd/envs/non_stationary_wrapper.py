from gym import Wrapper
from collections import deque
import numpy as np


class NonStationaryEnv(Wrapper):

    def __init__(self, env, change_freq=40000, tasks=['normal', 'joint-malfunction', 'wind', 'velocity']):
        super(NonStationaryEnv, self).__init__(env)
        self.tasks = deque(tasks)
        self.change_freq = change_freq
        self.action_dim = self.env.action_space.shape[0]
        self.malfunction_mask = np.ones(self.action_dim)
        self.malfunction_mask[0] = -1
        self.malfunction_mask[1] = -1
        self.counter = 0
        self.default_target_vel = 1.5
        self.fix_reward_vel = self.unwrapped.spec.id.startswith('Hopper') or self.unwrapped.spec.id.startswith('HalfCheetah')
        self.wind = [-4,0,0,0,0,0]  #-4
        self.no_wind = [0,0,0,0,0,0]

    @property
    def current_task(self):
        return self.tasks[0]

    def step(self, action):
        target_vel = self.default_target_vel
        posbefore = self.unwrapped.sim.data.qpos[0]

        if self.counter % self.change_freq == 0 and self.counter > 0:
            self.tasks.rotate(-1)
            print("CHANGED TO TASK {} AT STEP {}!".format(self.current_task, self.counter))
        
        if self.counter == 320000:
            self.tasks = deque(['wind', 'normal', 'velocity', 'joint-malfunction'])

        if self.current_task == 'normal':
            pass
        elif self.current_task == 'joint-malfunction':
            action = self.malfunction_mask*action
        elif self.current_task == 'velocity':
            target_vel = 2.0
        if self.current_task == 'wind':
            force = self.wind
        else:
            force = self.no_wind

        for part in self.unwrapped.sim.model._body_name2id.values():
            self.unwrapped.sim.data.xfrc_applied[part,:] = force

        next_obs, reward, done, info = self.env.step(action)
        info['task'] = self.current_task

        if self.fix_reward_vel:
            posafter = self.unwrapped.sim.data.qpos[0]
            forward_vel = (posafter - posbefore) / self.unwrapped.dt
            reward -= forward_vel # remove this term

            reward += -1 * abs(forward_vel - target_vel)

        self.counter += 1

        return next_obs, reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.trackbodyid = 0