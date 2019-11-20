# Run 10 subprocesses that run some number of rollouts.
# TODO: Pass in number of rollouts to run.
for i in `seq 1 10`;
do
  xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python extract.py &
  sleep 1.0
done
