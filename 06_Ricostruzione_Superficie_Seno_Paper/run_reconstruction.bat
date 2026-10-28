@echo off
setlocal
cd /d "%~dp0"
title Ricostruzione 3D Superficie Seno da Mammografie 2D

echo ======================================================================
echo  RICOSTRUZIONE 3D SUPERFICIE MAMMELLA DA COPPIA CC + MLO
echo  Implementazione scientifica (Huang et al. 2006 / Pinto et al. 2023)
echo ======================================================================
echo.

set "DEFAULT_PID=P_00001"
set "PATIENT_ID=%~1"

if "%PATIENT_ID%"=="" (
    set /p "PATIENT_ID=Inserisci l'ID del paziente (es. P_00081 o 81) [Default: %DEFAULT_PID%]: "
)

if "%PATIENT_ID%"=="" set "PATIENT_ID=%DEFAULT_PID%"
set "PATIENT_ID=%PATIENT_ID:"=%"
set "PATIENT_ID=%PATIENT_ID: =%"

set "DATA_DIR=..\data"
if not exist "%DATA_DIR%" set "DATA_DIR=data"

echo.
echo Avvio elaborazione per il paziente: %PATIENT_ID%...
python main.py --patient_id "%PATIENT_ID%" --data_dir "%DATA_DIR%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERRORE] Si e verificato un problema durante la ricostruzione.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ======================================================================
echo  Vuoi aprire il modello generato in Blender adesso? (S/N)
echo ======================================================================
set /p "OPEN_CHOICE=[S/N, Default: S]: "
if /i "%OPEN_CHOICE%"=="" set "OPEN_CHOICE=S"

if /i "%OPEN_CHOICE%"=="S" (
    call open_in_blender.bat "%PATIENT_ID%"
)

pause
