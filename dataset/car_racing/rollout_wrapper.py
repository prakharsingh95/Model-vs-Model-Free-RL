#!/usr/bin/env python3

from subprocess import Popen


def main():
    processes = []
    for i in range(8):
        args = ['xvfb-run', '-a', '-s', '-screen 0 1400x900x24 +extension RANDR',
                'python', 'rollout.py', '--num_rollouts=100', '--policy=controller']
        process = Popen(args)
        processes.append(process)
    for process in processes:
        process.wait()


if __name__ == '__main__':
    main()
