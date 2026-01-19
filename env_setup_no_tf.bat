@echo off
REM Install PyTorch

REM timm
pip install timm==0.9.12 --no-cache-dir

REM General Computer Vision Utils (Removing VTAB/TensorFlow specific packages)
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

echo Setup complete (TensorFlow excluded).
pause
