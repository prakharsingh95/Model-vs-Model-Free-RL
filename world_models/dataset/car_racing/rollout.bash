#!/usr/bin/env bash

function main() {
  # TODO: Figure out how to cancel this through <ctrl-C>
  # Set number of processes to run in parallel
  num_processes=10
  i=1
  while [ "$i" -le "$num_processes" ]; do
    (xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR"\
     -- python rollout.py --num_rollouts=1 &)
    i=$(($i + 1))
    sleep 1
  done
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
