@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo  FNN Diagnosis GUI - Windows EXE Builder
echo ============================================================

if not exist "diagnosis_gui.py" (
    echo [ERROR] diagnosis_gui.py 파일이 없습니다.
    pause
    exit /b 1
)

if not exist "fnn_fault_checkpoint.pt" (
    echo [ERROR] fnn_fault_checkpoint.pt 파일이 없습니다.
    echo Colab에서 생성한 PT 파일을 이 BAT 파일과 같은 폴더에 넣어주세요.
    pause
    exit /b 1
)

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python Launcher가 없습니다.
    echo Python 3.10 또는 3.11 설치 후 다시 실행해 주세요.
    pause
    exit /b 1
)

echo [1/3] 가상환경을 준비합니다...
if not exist ".venv\Scripts\python.exe" (
    py -m venv .venv
    if errorlevel 1 goto :build_error
)

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"

echo [2/3] PyTorch, PyQt5, PyInstaller를 설치합니다...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto :build_error
"%PYTHON_EXE%" -m pip install numpy torch PyQt5 pyinstaller
if errorlevel 1 goto :build_error

echo [3/3] 단일 EXE 파일을 생성합니다...
"%PYTHON_EXE%" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --name FNN_Diagnosis_GUI ^
    --collect-all torch ^
    --hidden-import PyQt5.sip ^
    --add-data "fnn_fault_checkpoint.pt;." ^
    diagnosis_gui.py

if errorlevel 1 goto :build_error

echo.
echo ============================================================
echo  EXE 생성 완료
echo  결과: %CD%\dist\FNN_Diagnosis_GUI.exe
echo ============================================================
start "" "%CD%\dist"
pause
exit /b 0

:build_error
echo.
echo [ERROR] EXE 생성 중 오류가 발생했습니다.
echo 위 오류 메시지를 확인해 주세요.
pause
exit /b 1
