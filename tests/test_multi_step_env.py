import unittest

import gym
import cv2

import numpy as np
import itertools as it

from utils import MultiStepEnv, VideoLogger, ReplayBuffer

from pympler import muppy, summary

class TestMultiStepEnv(unittest.TestCase):

    def setUp(self):
        self.height, self.width = 64, 64
        discrete_actions = ([-1, 0, 1], [1, 0], [0.2, 0])
        self.action_dict = {idx:np.array(action) for idx, action in enumerate(it.product(*discrete_actions))}
        self.mse = MultiStepEnv('CarRacing-v0', height = self.height, width=self.width, crop_height=84, frame_stack_size=4, frame_skips=1,action_dict=self.action_dict, max_neg_reward_steps=12)

    def test_step(self):
        # self.mse.reset(video_log_file='test_multi_step_env_from_env')

        
        # vl = VideoLogger('test_multi_step_env', self.height, self.width) 

        replay_memory = ReplayBuffer(capacity = 40000)
    
        mem_size = 0.0
        for eps in range(50):
            obs = self.mse.reset()
            total_reward = 0.0
            done = False
            n = 0

            # all_objects = muppy.get_objects()
            # sum1 = summary.summarize(all_objects)
            # summary.print_(sum1)


            while not done:
                action_idx = np.random.randint(len(self.action_dict))
                next_obs, reward, done, _ = self.mse.step(action_idx)

                total_reward += reward

                n += 1

                replay_memory.insert(obs, action_idx, reward, next_obs, done)

                mem_size += obs.nbytes + next_obs.nbytes

                # print(obs.shape, obs.dtype, obs.nbytes / (1024.0 * 1024.0))

                # print(f'n = {n}, total_reward = {total_reward}, done = {done}, action = {action}')
                if done:
                    break

                obs = next_obs
            print(f'eps {eps} done...', replay_memory.size, mem_size / (1024.0 * 1024.0), 100*(mem_size/48) / (1024.0 * 1024.0*1024))


                # vl.write(cv2.cvtColor(obs[0,:,:], cv2.COLOR_GRAY2RGB))

                

        # self.assertEqual(n, 500)