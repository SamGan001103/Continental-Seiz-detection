@echo off
REM ---------------------------------------------------------------------------
REM Launch the seizure-review GUI.
REM Double-click this file, or run it from a terminal in the repo root.
REM Optionally pass an EDF path to auto-open it:  launch_gui.bat path\to\file.edf
REM
REM Finds the seiz36 interpreter by searching, in order:
REM   1. %SEIZ36_PYTHON%          - explicit override, wins over everything
REM   2. %CONDA_PREFIX%           - an already-activated conda env
REM   3. %USERPROFILE%\miniconda3 / anaconda3  (and ProgramData installs)
REM   4. python on PATH
REM ---------------------------------------------------------------------------
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PYEXE="

REM --- 1. explicit override -------------------------------------------------
if defined SEIZ36_PYTHON (
  if exist "%SEIZ36_PYTHON%" (
    set "PYEXE=%SEIZ36_PYTHON%"
  ) else (
    echo SEIZ36_PYTHON is set but does not exist:
    echo   %SEIZ36_PYTHON%
    echo.
    pause
    exit /b 1
  )
)

REM --- 2. an activated conda environment ------------------------------------
if not defined PYEXE if defined CONDA_PREFIX (
  if exist "%CONDA_PREFIX%\python.exe" set "PYEXE=%CONDA_PREFIX%\python.exe"
)

REM --- 3. common install locations ------------------------------------------
if not defined PYEXE (
  for %%R in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\AppData\Local\miniconda3"
    "%ProgramData%\miniconda3"
    "%ProgramData%\Anaconda3"
    "C:\miniconda3"
    "C:\Anaconda3"
  ) do (
    if not defined PYEXE if exist "%%~R\envs\seiz36\python.exe" set "PYEXE=%%~R\envs\seiz36\python.exe"
  )
)

REM --- 4. whatever python is on PATH ----------------------------------------
if not defined PYEXE (
  for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PYEXE set "PYEXE=%%P"
  )
)

if not defined PYEXE (
  echo.
  echo Could not find a Python interpreter for the seizure-review GUI.
  echo.
  echo Create the environment once with:
  echo     conda env create -f environment-seiz36.yml
  echo     conda activate seiz36
  echo.
  echo Then re-run this file, or set SEIZ36_PYTHON to the python.exe to use:
  echo     set SEIZ36_PYTHON=C:\path\to\envs\seiz36\python.exe
  echo.
  pause
  exit /b 1
)

echo Using: %PYEXE%
"%PYEXE%" -m gui.main %*
if errorlevel 1 pause
endlocal
