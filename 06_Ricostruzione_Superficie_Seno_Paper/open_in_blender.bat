@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Apertura Modello 3D Seno in Blender

set "DEFAULT_PID=P_00001"
set "PATIENT_ID=%~1"

if "%PATIENT_ID%"=="" (
    set /p "PATIENT_ID=Inserisci l'ID del paziente da visualizzare (es. P_00081 o 81) [Default: %DEFAULT_PID%]: "
)

if "%PATIENT_ID%"=="" set "PATIENT_ID=%DEFAULT_PID%"
set "PATIENT_ID=%PATIENT_ID:"=%"
set "PATIENT_ID=%PATIENT_ID: =%"

rem Normalizzazione ID numerico se breve (es. 81 -> P_00081)
if exist "output\%PATIENT_ID%" (
    set "RESOLVED_PID=%PATIENT_ID%"
) else (
    rem Cerca cartelle corrispondenti in output
    set "RESOLVED_PID=%PATIENT_ID%"
    for /d %%D in (output\*!PATIENT_ID!*) do (
        set "RESOLVED_PID=%%~nxD"
    )
)

set "BLENDER_EXE=C:\Users\Pierluca\OneDrive\Documenti\Blender\blender.exe"
if not exist "%BLENDER_EXE%" set "BLENDER_EXE=..\Blender\blender.exe"
if not exist "%BLENDER_EXE%" set "BLENDER_EXE=blender"

set "OUT_DIR=output\%RESOLVED_PID%"
set "SCENE_FILE=%OUT_DIR%\%RESOLVED_PID%_scene.blend"
set "GLB_COMPOSITE=%OUT_DIR%\%RESOLVED_PID%_breast_with_tumor.glb"
set "GLB_SKIN=%OUT_DIR%\%RESOLVED_PID%_breast_surface.glb"

if exist "%SCENE_FILE%" (
    set "TARGET_FILE=%SCENE_FILE%"
) else if exist "%GLB_COMPOSITE%" (
    set "TARGET_FILE=%GLB_COMPOSITE%"
) else if exist "%GLB_SKIN%" (
    set "TARGET_FILE=%GLB_SKIN%"
) else (
    echo.
    echo [ERRORE] Nessun modello trovato per il paziente %RESOLVED_PID% in %OUT_DIR%!
    echo Esegui prima 'run_reconstruction.bat' per generare i modelli 3D.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo  Apertura in Blender
echo  Paziente: %RESOLVED_PID%
echo  File:     %TARGET_FILE%
echo ======================================================================
echo.

start "" "%BLENDER_EXE%" "%TARGET_FILE%"
