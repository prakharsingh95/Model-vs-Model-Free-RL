# Sample Efficient RL

### Demos

#### DQN

CarRacing-v0            |  VizdoomDefendCenter-v0
:-------------------------:|:-------------------------:
<img src="demos/dqn-CarRacing-v0.gif" height="300" width="300" style="display:inline;">  |  <img src="demos/dqn-VizdoomDefendCenter-v0.gif" height="300" width="300" style="display:inline;">


### Training World Models

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

### Training DQN

```
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" python3 train_dqn.py --task "CarRacing-v0"
```

Note that `xvfb-run` is necessary iff you are training on computer without a display connected (e.g. over SSH).
