#!/usr/bin/env bash

function main() {
  for i in $(seq 1)
  do
    (xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR"\
     -- python rollout.py &)
  done
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
