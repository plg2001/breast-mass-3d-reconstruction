@echo off
title Apertura Lesione 3D in Blender (Per Paziente)
echo ===============================================================================
echo   Apertura Modello 3D Ricostruito in Blender...
echo ===============================================================================
cd /d "%~dp0"

set BLENDER_EXE=C:\Users\Pierluca\OneDrive\Documenti\Blender\blender.exe
if not exist "%BLENDER_EXE%" set BLENDER_EXE=..\Blender\blender.exe

if not exist "%BLENDER_EXE%" (
    echo [ERRORE] Eseguibile Blender non trovato in: %BLENDER_EXE%
    pause
    exit /b
)

echo Scegli quale paziente visualizzare in Blender:
echo   [1] Paziente P_00232
echo   [2] Paziente P_01754
echo.
set /p SCELTA="Inserisci opzione (1 o 2, default: 1): "

set CASE_NAME=P_00232
if "%SCELTA%"=="2" set CASE_NAME=P_01754

set FILE_TO_OPEN=%~dp0output\%CASE_NAME%\meshes\lesion_reconstructed.blend
if not exist "%FILE_TO_OPEN%" set FILE_TO_OPEN=%~dp0output\%CASE_NAME%\meshes\lesion_reconstructed.glb

if exist "%FILE_TO_OPEN%" (
    echo Avvio Blender per il caso %CASE_NAME%: %FILE_TO_OPEN%
    start "" "%BLENDER_EXE%" "%FILE_TO_OPEN%"
) else (
    echo [AVVISO] Nessun modello trovato in %FILE_TO_OPEN%
    echo Esegui prima la ricostruzione con run_reconstruction.bat!
    pause
)
