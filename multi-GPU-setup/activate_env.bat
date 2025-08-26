@echo off
echo Activating Python 3.9 Multi-GPU Environment...
echo.
echo This environment includes:
echo - PyTorch 2.7.1 with CUDA 11.8 support
echo - Multi-GPU training capabilities
echo - Complete ML stack (numpy, scipy, matplotlib, pandas, scikit-learn)
echo - Jupyter notebooks and TensorBoard
echo.
echo Available scripts:
echo - test_gpus.py - Test your GPU setup
echo - multi_gpu_example_fixed.py - Multi-GPU training example
echo - gpu_monitor.py - Monitor GPU usage and performance
echo.
call venv\Scripts\activate.bat
echo.
echo Environment activated! You can now run:
echo   python test_gpus.py
echo   python multi_gpu_example_fixed.py
echo   python gpu_monitor.py
echo.
echo To deactivate, type: deactivate
echo.
