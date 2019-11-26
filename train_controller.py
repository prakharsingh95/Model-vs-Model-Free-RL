#!/usr/bin/env python3
'''Take heavy inspiration from the ctallec implementation.

Since we're minimizing an objective function, we need to invert our
rewards.
'''

import argparse
import cma
import os
import sys
import torch

from pathlib import Path
from torch.multiprocessing import Process, Queue

from controller import Controller

cwd = Path(os.path.dirname(__file__))
results = cwd/'results'
logdir = results/'controller'

# TODO: How to refactor these global variables? Can encapsulate them in
# an object.
p_queue = Queue()
r_queue = Queue()
e_queue = Queue()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=Path, default=logdir,
                        help='Where everything is stored.')
    parser.add_argument('--n-samples', type=int,
                        help='Number of samples used to obtain return'
                             'estimate.')
    parser.add_argument('--pop-size', type=int, help='Population size.')
    parser.add_argument('--target-return', type=float,
                        help='Stops once the return gets above target_return.')
    parser.add_argument('--display', action='store_true',
                        help='Use progress bars if specified.')
    parser.add_argument('--max-workers', type=int, default=1,
                        help='Maximum number of workers.')
    return parser.parse_args()


def flatten_parameters(params):
    # TODO: Move this to utils file
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()


def slave_routine(p_queue, r_queue, e_queue, p_index, logdir):
    """ Thread routine.

    Threads interact with p_queue, the parameters queue, r_queue, the
    result queue and e_queue the end queue. They pull parameters from
    p_queue, execute the corresponding rollout, then place the result in
    r_queue.

    Each parameter has its own unique id. Parameters are pulled as
    tuples (s_id, params) and results are pushed as (s_id, result).  The
    same parameter can appear multiple times in p_queue, displaying the
    same id each time.

    As soon as e_queue is non empty, the thread terminate.

    When multiple gpus are involved, the assigned gpu is determined by
    the process index p_index (gpu = p_index % n_gpus).

    :args p_queue: queue containing couples (s_id, parameters) to evaluate
    :args r_queue: where to place results (s_id, results)
    :args e_queue: as soon as not empty, terminate
    :args p_index: the process index
    """
    tmp_dir = logdir/'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # init routine
    gpu = p_index % torch.cuda.device_count()
    device = torch.device('cuda:{}'.format(gpu)
                          if torch.cuda.is_available() else 'cpu')

    # redirect streams
    sys.stdout = open(os.path.join(tmp_dir, str(os.getpid()) + '.out'), 'a')
    sys.stderr = open(os.path.join(tmp_dir, str(os.getpid()) + '.err'), 'a')

    with torch.no_grad():
        r_gen = RolloutGenerator(args.logdir, device, time_limit)

        while e_queue.empty():
            if p_queue.empty():
                sleep(.1)
            else:
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))


def run(args):
    latent = 32
    mixture = 256
    size = latent + mixture
    controller = Controller(size)

    Process(target=slave_routine,
            args=(p_queue, r_queue, e_queue, 0, args.logdir)).start()

    cur_best = None
    savefile = args.logdir/'best.tar'
    if savefile.exists():
        print(f'Loading from {savefile}')
        state = torch.load(savefile.as_posix(), map_location={'cuda:0': 'cpu'})
        cur_best = -state['reward']
        controller.load_state_dict(state['state_dict'])


    parameters = controller.parameters()
    es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1,
                                  {'popsize': args.pop_size})

    epoch = 0

    # TODO: Add training loop


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
