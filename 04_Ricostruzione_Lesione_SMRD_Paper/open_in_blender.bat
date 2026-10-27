@echo off
title Apertura Lesione SMRD 3D con Spikes in Blender
echo ===============================================================================
echo   Apertura Modello SMRD 3D Spicolato (De Sisternes et al. 2015) in Blender...
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
echo   [1] Paziente P_00001 (Paziente 1)
echo   [2] Paziente P_00081 (Paziente 81)
echo   [3] Paziente P_00232 (Caso Benchmark 1)
echo   [4] Paziente P_01226 (Caso con Spikes e Frastagliature)
echo   [5] Paziente P_01754
echo   Oppure scrivi direttamente l'ID del paziente (es. P_00005)
echo.
set /p SCELTA="Inserisci opzione o ID (default: 1): "

if "%SCELTA%"=="" set SCELTA=1
set CASE_NAME=%SCELTA%
if "%SCELTA%"=="1" set CASE_NAME=P_00001
if "%SCELTA%"=="2" set CASE_NAME=P_00081
if "%SCELTA%"=="3" set CASE_NAME=P_00232
if "%SCELTA%"=="4" set CASE_NAME=P_01226
if "%SCELTA%"=="5" set CASE_NAME=P_01754

set FILE_TO_OPEN=%~dp0output\%CASE_NAME%\%CASE_NAME%_smrd_lesion.blend
if not exist "%FILE_TO_OPEN%" set FILE_TO_OPEN=%~dp0output\%CASE_NAME%\meshes\lesion_reconstructed.blend
if not exist "%FILE_TO_OPEN%" set FILE_TO_OPEN=%~dp0output\%CASE_NAME%\%CASE_NAME%_smrd_lesion.glb
if not exist "%FILE_TO_OPEN%" set FILE_TO_OPEN=%~dp0output\%CASE_NAME%\%CASE_NAME%_smrd_lesion.obj

if exist "%FILE_TO_OPEN%" (
    echo Avvio Blender per il caso %CASE_NAME%: %FILE_TO_OPEN%
    if "%FILE_TO_OPEN:~-6%"==".blend" (
        start "" "%BLENDER_EXE%" "%FILE_TO_OPEN%"
    ) else (
        start "" "%BLENDER_EXE%" --python-expr "import bpy; bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=r'%FILE_TO_OPEN%'); [bpy.ops.object.shade_smooth() for o in bpy.context.scene.objects if o.type=='MESH']"
    )
) else (
    echo [AVVISO] Nessun modello trovato per %CASE_NAME% in: %FILE_TO_OPEN%
    echo Esegui prima la ricostruzione con smrd_lesion_reconstruction.py!
    pause
)
