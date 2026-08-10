@echo off
REM ===================================================================
REM  Build the distributable Seizure Review application.
REM
REM  Run this on a DEVELOPMENT machine that has the seiz36 environment.
REM  It produces dist\SeizureReview\ — the folder you copy to a hospital PC.
REM
REM      packaging\build_app.bat
REM ===================================================================
setlocal

set "REPO=%~dp0.."
pushd "%REPO%"

set "PYEXE=%USERPROFILE%\miniconda3\envs\seiz36\python.exe"
if not exist "%PYEXE%" set "PYEXE=C:\Users\User\miniconda3\envs\seiz36\python.exe"
if not exist "%PYEXE%" (
  echo [X] Could not find the seiz36 environment.
  echo     Run setup.bat first, or edit PYEXE at the top of this script.
  goto :fail
)

echo.
echo [1/4] Checking the model weights are present and unmodified...
"%PYEXE%" -c "import sys;sys.path.insert(0,'.');import eval_config as c;from gui.io.cache import sha256_file;h=sha256_file(c.WEIGHTS);sys.exit(0 if h==c.WEIGHTS_SHA256 else 1)"
if errorlevel 1 (
  echo [X] Model weights missing or hash mismatch. Refusing to build.
  echo     A build with the wrong weights would be indistinguishable from a
  echo     correct one at run time, so this check is fatal, not a warning.
  goto :fail
)
echo     OK

echo.
echo [2/4] Running the test suite...
"%PYEXE%" -m unittest discover -s tests -q
if errorlevel 1 (
  echo [X] Tests failed. Refusing to build a release from a failing tree.
  goto :fail
)
echo     OK

echo.
echo [3/4] Freezing with PyInstaller. This takes several minutes...
"%PYEXE%" -m PyInstaller packaging\SeizureReview.spec --noconfirm --distpath dist --workpath build\pyi
if errorlevel 1 goto :fail

echo.
echo [4/4] Smoke-testing the frozen application...
"%PYEXE%" packaging\smoke_test.py
if errorlevel 1 (
  echo [X] The frozen application failed its smoke test. Do not ship it.
  goto :fail
)

echo.
echo ============================================================
echo  Build complete:  %REPO%\dist\SeizureReview\
echo.
echo  To deploy: copy that whole folder to the target PC and run
echo  SeizureReview.exe. See docs\DEPLOYMENT.md.
echo ============================================================
popd
endlocal
exit /b 0

:fail
echo.
echo Build FAILED.
popd
endlocal
exit /b 1
