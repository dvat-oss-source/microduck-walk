#!/usr/bin/env bash
# Fetch the third-party pieces this repo builds on into ./upstream (not redistributed here).
#   - pollen-robotics/microduck_rl: robot MJCF + meshes (meshes are CC BY-NC-SA 4.0), task code
#   - Rhoban/bam: XL330 actuator model
#   - mjlab 1.3.0 / rsl-rl-lib 5.0.1 wheels (scene.xml, reference source the env is checked against)
#   - Hugging Face pollen-robotics/microduck-policies + the 441tzs6d_3750 training config
set -euo pipefail
cd "$(dirname "$0")/.."
U=upstream; mkdir -p $U/deps/whl $U/deps/src $U/bundle441

[ -d $U/microduck_rl ] || git clone https://github.com/pollen-robotics/microduck_rl $U/microduck_rl
git -C $U/microduck_rl checkout -q bae39f5900b318f7e87fc485c30a31213cfa14eb      # used by demo/ and the cluster run
[ -d $U/rl_cb8561f6 ] || git -C $U/microduck_rl worktree add -q ../rl_cb8561f6 cb8561f6  # task version the duckmac env reproduces
cp demo/scene_ramp_steps_demo.xml $U/microduck_rl/src/mjlab_microduck/robot/microduck/

[ -d $U/deps/bam ] || git clone -q https://github.com/Rhoban/bam $U/deps/bam
git -C $U/deps/bam checkout -q 62bd8ce12154340be97e06f7f41a0ca8f116d967

python3 -m pip download -q --no-deps --only-binary=:all: --python-version 3.13 -d $U/deps/whl mjlab==1.3.0 rsl-rl-lib==5.0.1
for w in $U/deps/whl/*.whl; do python3 -m zipfile -e "$w" $U/deps/src; done

hf download pollen-robotics/microduck-policies --local-dir $U/policies --quiet
hf download pollen-robotics/microduck-training --local-dir $U/.hf_training --quiet \
  --include "bundles/velocity2/441tzs6d_3750/*.yaml" "bundles/velocity2/441tzs6d_3750/*.json" "bundles/velocity2/441tzs6d_3750/*.md"
cp $U/.hf_training/bundles/velocity2/441tzs6d_3750/* $U/bundle441/
echo "upstream ready in ./$U"
