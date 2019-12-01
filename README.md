# Sample Efficient RL

### Demos

#### DQN (CarRacing-v0)

<img src="demos/dqn-CarRacing-v0.gif" height="250" width="250">

### How to Train

#### Training VAE
  * Source set_pythonpath.bash
  * Go into the dataset/car_racing directory and run rollout.bash
  * Run make_csv.py in datasets/car_racing
  * Call train_vae.py
  * Sampled mappings of noise reconstructions are seen in results

Increase the number of rollouts in rollout.bash and rollout.py to generate more data. This currently
trains on the random action policy, so there isn't much variation in the road.

#### Training MDRNN
* Train VAE as above (duh!)
* Call train_mdrnn.py

Note that training the MDRNN requires that the VAE is well trained!
