'''Save observations and actions observed through a policy. These saved
actions can be used to train a VAE.
'''

import gym
import numpy as np
import os

from pathlib import Path

from model import make_model

MAX_FRAMES = 1000


def run_trial(seed, render_mode=False):
  np.random.seed(seed)
  model = make_model(load_model=False)
  model.make_env(render_mode, full_episode=True)
  # Initialize to a random policy
  model.init_random_model_params(stdev=np.random.rand()*0.01)
  model.reset()
  model.env.seed(seed)
  recording_obs = []
  recording_action = []
  obs = model.env.reset() # pixels
  done = False
  num_seen_frames = 0
  while not done and num_seen_frames < MAX_FRAMES:
    if render_mode:
      model.env.render('human')
    else:
      model.env.render('rgb_array')
    z, mu, logvar = model.encode_obs(obs)
    action = model.get_action(z)
    obs, reward, done, info = model.env.step(action)
    recording_action.append(action)
    recording_obs.append(obs)
    num_seen_frames += 1
  model.env.close()
  return (np.array(recording_obs, dtype=np.uint8),
          np.array(recording_action, dtype=np.float16))


def main():
  cwd = Path(os.path.dirname(__file__))
  savedir = cwd/'record'
  savedir.mkdir(exist_ok=True)
  num_trials = 88
  for i in range(num_trials):
    seed = np.random.randint(0, 2**31-1)
    observations, actions = run_trial(seed)
    filename = savedir/f'{seed}.npz'
    np.savez_compressed(filename, obs=observations, action=actions)


if __name__ == '__main__':
  main()
