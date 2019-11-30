import numpy as np

from PIL import Image

import gym

import settings

from .misc import VideoLogger

from random import sample

from skimage import color, transform

# from pympler import muppy, summary

class MultiStepEnv(object):

    def __init__(self, env_name: str, frame_stack_size=None, frame_skips=None, crop_height=None, crop_width=None, height=None, width=None, action_dict=None, max_neg_reward_steps=100):
        assert height is not None and width is not None
        assert frame_stack_size is not None and frame_stack_size >= 2
        assert frame_skips is not None

        self.env_name = env_name
        self.frame_stack_size = frame_stack_size
        self.frame_skips = frame_skips
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.max_neg_reward_steps = max_neg_reward_steps

        self.env = None
        self.buffer = None
        self.video_logger = None
        self.neg_reward_steps = None
        self.height, self.width = height, width
        self.action_dict = action_dict


    def reset(self, video_log_file=None):
        if self.env is None:
            self.env = gym.make(self.env_name)

        first_obs = self.process_frame(self.env.reset())
        # self.buffer = np.zeros((self.height,self.width, self.frame_stack_size)).astype(np.uint8)
        self.buffer = np.repeat(first_obs, self.frame_stack_size, axis=2)
        
        if video_log_file is not None:
            env_height, env_width, _ = self.env.observation_space.shape
            self.video_logger = VideoLogger(video_log_file, 
                env_height, env_width) 
        else:
            self.video_logger = None
        self.neg_reward_steps = 0

        return self.normalize(self.buffer).copy()

    @property
    def num_actions(self):
        return len(self.action_dict)

    @staticmethod
    def normalize(obs):
        # return (obs/255 * 2 - 1).astype(np.float32)
        return obs

    @property
    def gym_env(self):
        return self.env
    
    def process_frame(self, obs):
        # Crop
        if self.crop_height is not None:
            obs = obs[:self.crop_height, :, :]
        if self.crop_width is not None:
            obs = obs[:, self.crop_width, :]

        # Resize
        if obs.shape[0] != self.height or obs.shape[1] != self.width:
            obs = np.array(Image.fromarray(obs).resize((self.width, self.height), resample=Image.BICUBIC))

        # Convert to grayscale
        obs = color.rgb2gray(obs)
        # print(obs*2-1)
        obs = (obs * 255).astype(np.uint8)
        # print(obs/255.0 * 2 - 1)
        # print(obs.max(), obs.min())
        # raise "XYZ"
        # obs = np.array(Image.fromarray(obs).convert('L'))

        obs = np.expand_dims(obs, -1)

        return obs

    def random_action(self):
        gas_actions = np.array([1 if a[1] == 1 and a[2] == 0 else 0 for a in self.action_dict.values()])
        weights = 14.0 * gas_actions + 1.0
        p = weights / np.sum(weights)
        # print(p)
        action_idx = np.random.choice(np.arange(len(self.action_dict)), p=p)
        return action_idx

    def step(self, action_idx):
        total_reward = 0.0
        done = False

        action = self.action_dict[action_idx] 

        for _ in range(self.frame_skips):
            obs, reward, done, _ = self.env.step(action)
            # self.env.render()

            if self.video_logger is not None:
                self.video_logger.write(obs.copy())

            total_reward += reward

            if done:
                # obs = obs * 0
                break

        # if not done:
        if total_reward < 0.0:
            self.neg_reward_steps += 1
        else:
            self.neg_reward_steps = 0
        if self.neg_reward_steps == self.max_neg_reward_steps:
            done = True
            total_reward += -20.0

        obs = self.process_frame(obs).copy()

        self.buffer = np.concatenate((self.buffer[:,:,1:], obs), axis=-1).copy()

        return self.normalize(self.buffer).copy(), total_reward, done, {}
