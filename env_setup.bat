@echo off
REM Install PyTorch
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir

REM timm
pip install timm==0.9.12 --no-cache-dir

REM VTAB
pip install tensorflow==2.11.0 --no-cache-dir
REM specifying tfds versions is important to reproduce our results
pip install tfds-nightly==4.4.0.dev202201080107 --no-cache-dir
pip install tensorflow-addons==0.19.0 --no-cache-dir
pip install opencv-python --no-cache-dir

REM CLIP
pip install git+https://github.com/openai/CLIP.git --no-cache-dir

REM utils
pip install dotwiz --no-cache-dir
pip install pyyaml --no-cache-dir
pip install tabulate  --no-cache-dir
pip install termcolor --no-cache-dir
pip install iopath --no-cache-dir
pip install scikit-learn --no-cache-dir

pip install ftfy regex tqdm --no-cache-dir
pip install pandas --no-cache-dir
pip install matplotlib --no-cache-dir
pip install ipykernel --no-cache-dir

echo Setup complete.
pause
