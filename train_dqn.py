import numpy as np
import itertools as it
import argparse

from pathlib import Path

from dqn import DQN
from utils import MultiStepEnv
import settings

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train OpenAI gym agent using a Deep Q-Network')
    parser.add_argument('--task', type=str, default='CarRacing-v0')
    args = parser.parse_args()

    # Load default DQN parameters from settings
    DQN_TRAIN_BATCH_SIZE = settings.DQN_TRAIN_BATCH_SIZE
    DQN_FRAME_STACK_SIZE = settings.DQN_FRAME_STACK_SIZE
    DQN_STEPS_PER_UPDATE = settings.DQN_STEPS_PER_UPDATE
    DQN_FRAME_SKIPS = settings.DQN_FRAME_SKIPS
    DQN_GAMMA = settings.DQN_GAMMA
    DQN_EPS_MAX = settings.DQN_EPS_MAX
    DQN_EPS_MIN = settings.DQN_EPS_MIN
    DQN_EPS_DECAY_STEPS = settings.DQN_EPS_DECAY_STEPS
    DQN_REPLAY_MIN_SIZE = settings.DQN_REPLAY_MIN_SIZE
    DQN_REPLAY_MAX_SIZE = settings.DQN_REPLAY_MAX_SIZE
    DQN_TARGET_UPDATE_FREQ = settings.DQN_TARGET_UPDATE_FREQ
    DQN_OPTIM_LR = settings.DQN_OPTIM_LR
    DQN_OPTIM_L2_REG_COEFF = settings.DQN_OPTIM_L2_REG_COEFF
    DQN_ENABLE_RGB = settings.DQN_ENABLE_RGB
    DQN_EVAL_FREQ = settings.DQN_EVAL_FREQ
    DQN_RENDER_MODE = settings.DQN_RENDER_MODE
    DQN_WEIGHTS_SAVE_FILE = f'dqn-{args.task}.pt'

    # Modify DQN parameters depending on task
    if args.task == 'CarRacing-v0':
        discrete_actions = ([-1, 0, 1], [1, 0], [1, 0])
        action_dict = {
            idx: np.array(action)
            for idx, action in enumerate(it.product(*discrete_actions))
        }
        HEIGHT, WIDTH = 96, 96
        DQN_EPS_MAX = 0.1
    elif args.task == 'VizdoomBasic-v0':
        action_dict = {idx: idx for idx in range(3)}
        HEIGHT, WIDTH = 96, 96
    elif args.task == 'LunarLander-v2':
        action_dict = {idx: idx for idx in range(4)}
        HEIGHT, WIDTH = 96, 128
        RENDER_MODE = 'indirect'
        FRAME_SKIPS = 0
        EPS_MAX = 1.0
        EPS_DECAY_STEPS = 1000000
        REPLAY_MAX_SIZE = 100000
    elif args.task == 'Breakout-v0':
        action_dict = {idx: idx for idx in range(4)}
        HEIGHT, WIDTH = 128, 96
        FRAME_SKIPS = 2
        EPS_MAX = 1.0
        EPS_DECAY_STEPS = 200000
        REPLAY_MAX_SIZE = 40000
    else:
        raise NotImplementedError
>>>>>>> dqn

    env = MultiStepEnv(
        args.task,
        frame_stack_size=DQN_FRAME_STACK_SIZE,
        frame_skips=DQN_FRAME_SKIPS,
        height=HEIGHT,
        width=WIDTH,
        action_dict=action_dict, 
        enable_rgb=DQN_ENABLE_RGB,
        render_mode=DQN_RENDER_MODE
    )

    dqn = DQN(
        multi_step_env = env,
        gamma = DQN_GAMMA,
        eps_max = DQN_EPS_MAX,
        eps_min = DQN_EPS_MIN,
        eps_decay_steps = DQN_EPS_DECAY_STEPS,
        replay_min_size = DQN_REPLAY_MIN_SIZE,
        replay_max_size = DQN_REPLAY_MAX_SIZE,
        target_update_freq = DQN_TARGET_UPDATE_FREQ,
        train_batch_size = DQN_TRAIN_BATCH_SIZE,
        steps_per_update = DQN_STEPS_PER_UPDATE,
        enable_rgb=DQN_ENABLE_RGB,
        model_save_file=DQN_WEIGHTS_SAVE_FILE,
        optim_lr = DQN_OPTIM_LR,
        optim_l2_reg_coeff = DQN_OPTIM_L2_REG_COEFF,
        eval_freq = DQN_EVAL_FREQ
    )

    if Path(DQN_WEIGHTS_SAVE_FILE).exists():
        print(f'Loading from checkpoint {DQN_WEIGHTS_SAVE_FILE}...')
        dqn.load_state(DQN_WEIGHTS_SAVE_FILE)
    else:
        print(f'Checkpoint {DQN_WEIGHTS_SAVE_FILE} does not exist...')

    dqn.eval(1)
    dqn.train(2000)
    dqn.test(100)
