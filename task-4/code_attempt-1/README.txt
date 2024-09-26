SETUP
=====
1. Install relevant Torch version : pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
2. Install detectron : (link : https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# On macOS, you may need to prepend the above commands with a few environment variables:
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install ...
```
