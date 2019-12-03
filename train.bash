#!/usr/bin/env bash

function rebuild_rollouts() {
  cd dataset/car_racing
  rm -rf rollouts
  python rollout_wrapper.py
  python make_csv.py
  cd ../..
}

function main() {
  i=1
  num_iters=2
  while [ "$i" -le "$num_iters" ]
  do
    rebuild_rollouts

    timeout 7200s python train_vae.py
    timeout 7200s python train_mdrnn.py
    rm controller.pt
    rm results/controller/best.tar
    timeout 18000s python train_controller.py --n-samples 4 --pop-size 6\
                          --target-return 950 --max-workers=12

    cp vae.pt "vae$i.pt"
    cp mdrnn.pt "mdrnn$i.pt"
    cp controller.pt "controller$i.pt"
    cp results/controller/best.tar "best$i.tar"

    i=$(($i + 1))
  done
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
