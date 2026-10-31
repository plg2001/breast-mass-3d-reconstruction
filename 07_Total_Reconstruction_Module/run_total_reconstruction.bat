@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Ricostruzione 3D Totale Mammella + Lesione SMRD

echo ===============================================================================
echo   RICOSTRUZIONE 3D TOTALE: MAMMELLA A 14 STRATI + LESIONE TUMORALE SMRD
echo   Integrazione Modulo 04 (SMRD) + Modulo 05 (Superficie) + Modulo 06 (AI U-Net)
echo ===============================================================================
echo.

:: Rilevamento interprete Python con PyTorch CUDA
set "PYTHON_CMD="
if exist "C:\Python311\python.exe" (
    set "PYTHON_CMD=C:\Python311\python.exe"
) else (
    py -3.11 -c "import torch" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=py -3.11"
    ) else (
        set "PYTHON_CMD=python"
    )
)

echo Interprete Python in uso: %PYTHON_CMD%
echo.

set "DEFAULT_PID=P_00081"
set "TARGET_PID=%~1"

if "%TARGET_PID%"=="" (
    set /p "TARGET_PID=Inserisci l'ID del paziente [Default: %DEFAULT_PID%]: "
)

if "%TARGET_PID%"=="" set "TARGET_PID=%DEFAULT_PID%"
set "TARGET_PID=%TARGET_PID:"=%"
set "TARGET_PID=%TARGET_PID: =%"

echo.
echo ===============================================================================
echo Avvio RICOSTRUZIONE 3D TOTALE per paziente: %TARGET_PID%
echo (Mammella biologica a 14 strati + Tumore spicolato SMRD localizzato in 3D)
echo ===============================================================================
echo.

"%PYTHON_CMD%" src\total_reconstruction.py --patient_id "%TARGET_PID%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERRORE] Si e verificato un problema durante la ricostruzione totale.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===============================================================================
echo  Vuoi aprire la scena totale 3D in Blender adesso? (S/N)
echo ===============================================================================
set /p "OPEN_CHOICE=[S/N, Default: S]: "
if /i "%OPEN_CHOICE%"=="" set "OPEN_CHOICE=S"

if /i "%OPEN_CHOICE%"=="S" (
    call open_in_blender.bat "%TARGET_PID%"
)

pause
