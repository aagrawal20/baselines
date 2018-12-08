import os

import numpy as np

import baselines.contract
import gym
from baselines.contract.bench.step_monitor import LogBuffer


class ConstraintEnv(gym.Wrapper):
    def __init__(self,
                 env,
                 constraints,
                 augmentation_type=None,
                 log_dir=None):
        gym.Wrapper.__init__(self, env)
        self.constraints = constraints
        self.augmentation_type = augmentation_type
        if log_dir is not None:
            self.log_dir = log_dir
            self.viol_log_dict = dict([(c, LogBuffer(1000, (), dtype=np.bool))
                             for c in constraints])
            self.rew_mod_log_dict = dict([(c, LogBuffer(1000, (), dtype=np.float32))
                             for c in constraints])
        else:
            self.logs = None

    def reset(self, **kwargs):
        [c.reset() for c in self.constraints]
        [
            log.save(os.path.join(self.log_dir, c.name + '_viols'))
            for (c, log) in self.viol_log_dict.items()
        ]
        [
            log.save(os.path.join(self.log_dir, c.name + '_rew_mod'))
            for (c, log) in self.rew_mod_log_dict.items()
        ]

        ob = self.env.reset(**kwargs)
        if self.augmentation_type == 'contract_state':
            ob = np.array([ob, [c.state_id() for c in self.constraints]])
        return ob

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        for c in self.constraints:
            is_vio, rew_mod = c.step(action, done)
            rew += rew_mod
            if self.viol_log_dict is not None:
                self.viol_log_dict[c].log(is_vio)
                self.rew_mod_log_dict[c].log(rew_mod)

        if self.augmentation_type == 'contract_state':
            ob = np.array([ob, [c.state_id() for c in self.constraints]])

        return ob, rew, done, info

    def __del__(self):
        print("DELETE!!!!!!!")
        self.reset()