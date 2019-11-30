import numpy as np
import itertools as it

from pathlib import Path

from dqn import DQN
from utils import MultiStepEnv

import settings

if __name__ == '__main__':

    discrete_actions = ([-1, 0, 1], [1, 0], [0.2, 0])
    action_dict = {
        idx: np.array(action)
        for idx, action in enumerate(it.product(*discrete_actions))
    }

    env = MultiStepEnv(
        'CarRacing-v0',
        frame_stack_size=settings.DQN_FRAME_STACK_SIZE,
        frame_skips=settings.DQN_FRAME_SKIPS,
        crop_height=None,
        crop_width=None,
        height=96,
        width=96,
        action_dict=action_dict, 
        max_neg_reward_steps=settings.DQN_MAX_NEG_STEPS
    )

    dqn = DQN(
        multi_step_env = env,
        gamma = settings.DQN_GAMMA,
        eps_max = settings.DQN_EPS_MAX,
        eps_min = settings.DQN_EPS_MIN,
        eps_decay_steps = settings.DQN_EPS_DECAY_STEPS,
        replay_min_size = settings.DQN_REPLAY_MIN_SIZE,
        replay_max_size = settings.DQN_REPLAY_MAX_SIZE,
        target_update_freq = settings.DQN_TARGET_UPDATE_FREQ,
        train_batch_size = settings.DQN_TRAIN_BATCH_SIZE
    )

    # if Path(settings.DQN_WEIGHTS_SAVE_FILE).exists():
    #     print('Loading from checkpoint...')
    #     dqn.load_state(settings.DQN_WEIGHTS_SAVE_FILE)
    # else:
    #     print('No checkpoint found...')

    dqn.train(5000)
