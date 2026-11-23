#!/bin/sh
# Build the C++ batched stepper against the MuJoCo 3.10.0 shipped in the venv wheel.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MJ=$($PY -c "import mujoco,os;print(os.path.dirname(mujoco.__file__))")
EXT=$($PY -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
clang++ -O3 -mcpu=native -std=c++17 -fvisibility=hidden -shared -undefined dynamic_lookup \
  $($PY -m pybind11 --includes) -I"$MJ/include" \
  duckmac/csrc/batch.cpp -L"$MJ" -lmujoco.3.10.0 -Wl,-rpath,"$MJ" -Wl,-headerpad_max_install_names \
  -o "duckmac/_batch$EXT"
echo built duckmac/_batch$EXT
# The wheel's dylib has a framework install name; point at the real file.
install_name_tool -change @rpath/mujoco.framework/Versions/A/libmujoco.3.10.0.dylib "$MJ/libmujoco.3.10.0.dylib" "duckmac/_batch$EXT"
