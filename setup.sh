
pip install pynvml==12.0.0
pip install -U tensordict
export NCCL_TIMEOUT=36000

pip install latex2sympy2
pip install word2number
pip install pebble
pip install timeout-decorator
pip install math-verify[antlr4_9_3]
pip uninstall -y antlr4-python3-runtime
pip install antlr4-python3-runtime==4.9.3

pip install addftool  jsonlines math_verify tensorboardX


pip3.10 install -e ./verl-latest
pip3.10 install --no-deps -e .
pip uninstall -y sglang # uninstall sglang without custom patches
pip install -e ./sglang-0.4.9.post6/ # install sglang with custom patches
pip install -e ./transformers-4.54.0/ # install transformers with custom patches


