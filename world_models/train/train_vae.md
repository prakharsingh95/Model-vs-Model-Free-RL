In order to train the vae
  * Source set_pythonpath.bash
  * Go into the dataset/car_racing directory and run rollout.bash
  * Run make_csv.py in datasets/car_racing
  * Call train/train_vae.py
  * Sampled mappings of noise reconstructions are seen in results

Increase the number of rollouts in rollout.bash and rollout.py to generate more data. This currently
trains on the random action policy, so there isn't much variation in the road.