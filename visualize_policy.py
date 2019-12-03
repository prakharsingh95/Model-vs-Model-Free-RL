#!/usr/bin/env python3

import os

import settings
import utils.misc as misc

from pathlib import Path

cwd = Path(os.path.dirname(__file__))


def main():
    rg = misc.RolloutGenerator(mdir=cwd, device=settings.device,
                               time_limit=1000)
    rg.rollout(None, render=True)


if __name__ == '__main__':
    main()
