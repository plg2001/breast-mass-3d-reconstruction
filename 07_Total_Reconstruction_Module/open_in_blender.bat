@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Apertura Scena 3D Totale Mammella + Lesione in Blender

set "DEFAULT_PID=P_00081"
set "PATIENT_ID=%~1"

if "%PATIENT_ID%"=="" (
    set /p "PATIENT_ID=Inserisci l'ID del paziente (es. P_00081 o 81) [Default: %DEFAULT_PID%]: "
)

if "%PATIENT_ID%"=="" set "PATIENT_ID=%DEFAULT_PID%"
set "PATIENT_ID=%PATIENT_ID:"=%"
set "PATIENT_ID=%PATIENT_ID: =%"

if exist "output\%PATIENT_ID%" (
    set "RESOLVED_PID=%PATIENT_ID%"
) else (
    set "RESOLVED_PID=%PATIENT_ID%"
    for /d %%D in (output\*!PATIENT_ID!*) do (
        set "RESOLVED_PID=%%~nxD"
    )
)

set "BLENDER_EXE=C:\Users\Pierluca\OneDrive\Documenti\Blender\blender.exe"
if not exist "%BLENDER_EXE%" set "BLENDER_EXE=..\Blender\blender.exe"
if not exist "%BLENDER_EXE%" set "BLENDER_EXE=blender"

set "OUT_DIR=output\%RESOLVED_PID%"
set "SCENE_FILE=%OUT_DIR%\%RESOLVED_PID%_total_3d.blend"
set "GLB_FILE=%OUT_DIR%\%RESOLVED_PID%_total_3d.glb"

if exist "%SCENE_FILE%" (
    set "TARGET_FILE=%SCENE_FILE%"
) else if exist "%GLB_FILE%" (
    set "TARGET_FILE=%GLB_FILE%"
) else (
    echo.
    echo [ERRORE] Nessun modello totale trovato per il paziente %RESOLVED_PID% in %OUT_DIR%!
    echo Esegui prima 'run_total_reconstruction.bat' per generare la ricostruzione.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo  Apertura Scena 3D Totale in Blender (14 Strati Mammella + Tumore SMRD)
echo  Paziente: %RESOLVED_PID%
echo  File    : %TARGET_FILE%
echo ======================================================================
echo.

start "" "%BLENDER_EXE%" "%TARGET_FILE%"
