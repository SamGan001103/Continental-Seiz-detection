@echo off
REM ---------------------------------------------------------------------------
REM One-command setup for the seizure-review GUI.
REM Run this from an Anaconda Prompt in the project folder:  setup.bat
REM Creates the seiz36 environment, then verifies it can actually start.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   Seizure Review - environment setup
echo   This takes about 10 minutes and downloads ~2 GB.
echo ============================================================
echo.

where conda >nul 2>nul
if errorlevel 1 (
  echo [X] conda was not found.
  echo.
  echo     Install Miniconda from https://www.anaconda.com/download/success
  echo     then run this again from the "Anaconda Prompt" ^(Start menu^),
  echo     not from a normal PowerShell or Command Prompt window.
  echo.
  pause
  exit /b 1
)

REM --- does it already exist? ------------------------------------------------
conda env list | findstr /R /C:"^seiz36 " >nul
if not errorlevel 1 (
  echo [i] Environment "seiz36" already exists - skipping creation.
  goto verify
)

echo [1/2] Creating the seiz36 environment...
conda env create -f environment-seiz36.yml
if errorlevel 1 (
  echo.
  echo [X] Environment creation failed.
  echo.
  echo     The most common cause is that Python 3.6 packages can no longer be
  echo     resolved from the configured channels - it is end-of-life. See
  echo     INSTALL.md section 3 for what to do about that.
  echo.
  pause
  exit /b 1
)

:verify
echo.
echo [2/2] Verifying...
for /f "delims=" %%P in ('conda run -n seiz36 python -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
if not defined PYEXE (
  echo [X] Could not locate the seiz36 interpreter after creation.
  pause
  exit /b 1
)

"%PYEXE%" -c "import PyQt5, pyqtgraph, numpy, scipy, mne, tensorflow, sklearn; print('  dependencies OK')"
if errorlevel 1 (
  echo [X] The environment exists but a required package is missing.
  pause
  exit /b 1
)

"%PYEXE%" -c "import os,sys;sys.path.insert(0,'.');import eval_config as c;from gui.io.cache import sha256_file;h=sha256_file(c.WEIGHTS);print('  model weights OK' if h==c.WEIGHTS_SHA256 else '  WARNING: weights hash mismatch')"
if errorlevel 1 (
  echo [X] Could not verify the model weights. Is convlstm_ICA_12_train.h5 present?
  pause
  exit /b 1
)

"%PYEXE%" -c "import sys;sys.path.insert(0,'.');import gui.app;print('  GUI imports OK')"
if errorlevel 1 (
  echo [X] The GUI failed to import.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo   SETUP OK
echo.
echo   Start the application by double-clicking:  launch_gui.bat
echo.
echo   Note: EEG recordings are NOT included. See INSTALL.md.
echo ============================================================
echo.
pause
endlocal
