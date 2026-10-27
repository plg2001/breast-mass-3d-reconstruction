@echo off
setlocal
cd /d "%~dp0"
title Apertura Modello 3D Superficie Mammella in Blender

set DEFAULT_ID=P_00005
set PATIENT_ID=%~1

if "%PATIENT_ID%"=="" (
    echo ========================================================
    echo  Apertura Modello 3D in Blender
    echo ========================================================
    echo Esempi di ID disponibili: P_00001, P_00005, ecc.
    set /p "PATIENT_ID=Inserisci l'ID del paziente da visualizzare [Default: %DEFAULT_ID%]: "
)

if "%PATIENT_ID%"=="" set PATIENT_ID=%DEFAULT_ID%
set PATIENT_ID=%PATIENT_ID:"=%
set PATIENT_ID=%PATIENT_ID: =%

set BLENDER_EXE=C:\Users\Pierluca\OneDrive\Documenti\Blender\blender.exe
if not exist "%BLENDER_EXE%" set BLENDER_EXE=..\Blender\blender.exe

set MODEL_PATH=output\%PATIENT_ID%\breast_surface_scene.blend
if not exist "%MODEL_PATH%" set MODEL_PATH=output\%PATIENT_ID%\breast_skin_surface.glb
if not exist "%MODEL_PATH%" set MODEL_PATH=output\%PATIENT_ID%\breast_skin_surface.obj

if not exist "%MODEL_PATH%" (
    echo.
    echo [ERRORE] Nessun modello trovato per il paziente %PATIENT_ID%.
    echo Esegui prima 'run_surface_reconstruction.bat' per generare il modello 3D.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo  Apertura Modello 3D in Blender
echo  Paziente: %PATIENT_ID%
echo  File:     %MODEL_PATH%
echo ========================================================

if exist "%BLENDER_EXE%" (
    start "" "%BLENDER_EXE%" "%MODEL_PATH%"
) else (
    echo [AVVISO] Eseguibile Blender non trovato nel percorso predefinito.
    echo Apertura con l'applicazione 3D predefinita di Windows...
    start "" "%MODEL_PATH%"
)
