'''Take heavy inspiration from the ctallec implementation.

Since we're minimizing through CMA-ES, we need to multiply the received
reward by -1.
'''

import argparse
import cma
import numpy as np
import os
import sys
import torch

from pathlib import Path
from time import sleep
from torch.multiprocessing import Process, Queue
from tqdm import tqdm
from xvfbwrapper import Xvfb

from controller import Controller
from utils.misc import RolloutGenerator
from utils.misc import flatten_parameters
from utils.misc import load_parameters


cwd = Path(os.path.dirname(__file__))
results = cwd/'results'
logdir = results/'controller'
controller_pt = cwd/'controller.pt'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=Path, default=logdir,
                        help='Where everything is stored.')
    parser.add_argument('--n-samples', type=int, default=4,
                        help='Number of samples used to obtain return'
                             'estimate.')
    parser.add_argument('--pop-size', type=int, default=4, help='Population size.')
    parser.add_argument('--target-return', type=float,
                        help='Stops once the return gets above target_return.')
    parser.add_argument('--display', action='store_true',
                        help='Use progress bars if specified.')
    parser.add_argument('--max-workers', type=int, default=12,
                        help='Maximum number of workers.')
    return parser.parse_args()


def evaluate(solutions, results, p_queue, r_queue, rollouts=96):
    """ Give current controller evaluation.

    Evaluation is minus the cumulated reward averaged over rollout runs.

    :args solutions: CMA set of solutions
    :args results: corresponding results
    :args rollouts: number of rollouts

    :returns: minus averaged cumulated reward
    """
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    restimates = []

    for s_id in range(rollouts):
        p_queue.put((s_id, best_guess))

    print("Evaluating...")
    for _ in tqdm(range(rollouts)):
        while r_queue.empty():
            sleep(.1)
        restimates.append(r_queue.get()[1])

    return best_guess, np.mean(restimates), np.std(restimates)


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
    # Prevent subprocesses from displaying to main X server
    with Xvfb() as xvfb:
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
            r_gen = RolloutGenerator(Path('.'), device, time_limit=1000)

            while e_queue.empty():
                if p_queue.empty():
                    sleep(.1)
                else:
                    s_id, params = p_queue.get()
                    r_queue.put((s_id, r_gen.rollout(params)))


def run(args):
    p_queue = Queue()
    r_queue = Queue()
    e_queue = Queue()

    latent = 32
    mixture = 256
    size = latent + mixture
    controller = Controller(size, 3)

    for i in range(args.max_workers):
        Process(target=slave_routine,
                args=(p_queue, r_queue, e_queue, i, args.logdir)).start()

    cur_best = None
    savefile = args.logdir/'best.tar'
    if savefile.exists():
        print(f'Loading from {savefile}')
        state = torch.load(savefile.as_posix(), map_location={'cuda:0': 'cpu'})
        cur_best = -state['reward']
        controller.load_state_dict(state['state_dict'])


    parameters = controller.parameters()
    sigma = 0.1
    es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), sigma,
                                  {'popsize': args.pop_size})

    epoch = 0
    while not es.stop():
        if cur_best is not None and -cur_best > args.target_return:
            print('Already better than target, breaking...')
            break

        r_list = [0] * args.pop_size  # result list
        solutions = es.ask()

        # push parameters to queue
        for s_id, s in enumerate(solutions):
            for _ in range(args.n_samples):
                p_queue.put((s_id, s))

        # Retrieve results
        if args.display:
            pbar = tqdm(total=args.pop_size * args.n_samples)
        for _ in range(args.pop_size * args.n_samples):
            while r_queue.empty():
                sleep(.1)
            r_s_id, r = r_queue.get()
            r_list[r_s_id] += r / args.n_samples
            if args.display:
                pbar.update(1)
        if args.display:
            pbar.close()

        es.tell(solutions, r_list)
        es.disp()

        # CMA-ES seeks to minimize, so we want to multiply the reward we
        # get in a rollout by -1.

        best_params, best, std_best = evaluate(solutions, r_list, p_queue,
                                               r_queue)
        if (not cur_best) or (cur_best > best):
            cur_best = best
            print(f'Saving new best with value {-cur_best}+{-std_best}')
            load_parameters(best_params, controller)
            torch.save({'epoch': epoch,
                        'reward': -cur_best,
                        'state_dict': controller.state_dict()},
                       savefile)
            # Save after every epoch
            torch.save(controller.state_dict(), f'{controller_pt}')
        if -best > args.target_return:
            print(f'Terminating controller training with value {best}...')
            break
        epoch += 1

    es.result_pretty()
    e_queue.put('EOP')


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
