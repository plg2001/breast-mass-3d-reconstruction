@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Ricostruzione 3D Mammella e Tumore (3D U-Net-GAN)

echo ===============================================================================
echo   RICOSTRUZIONE 3D MAMMELLA E LOCALIZZAZIONE TUMORE (3D U-Net-GAN)
echo   Riferimento Paper: 3D Mammogram Reconstruction e Tumor Localization (2026)
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
set "MODE=UNIFIED"

if not "%TARGET_PID%"=="" goto DO_RUN

:MENU
echo Scegli la modalita di ricostruzione:
echo   [1] Ricostruzione UNIFICATA: Superficie Anatomica (Modulo 06) + Tumore U-Net-GAN (Modulo 07) [CONSIGLIATA]
echo   [2] Ricostruzione 3D U-Net-GAN pura (singola vista con griglia a 10 slice)
echo   [3] Test rapido con mammografia sintetica
echo.
set /p "CHOICE=Scegli opzione [1-3, Default: 1]: "
if "%CHOICE%"=="" set "CHOICE=1"

if "%CHOICE%"=="3" (
    set "TARGET_PID=SYNTHETIC"
    set "MODE=SYNTHETIC"
    goto DO_RUN
)
if "%CHOICE%"=="2" (
    set "MODE=STANDALONE"
) else (
    set "MODE=UNIFIED"
)

echo.
set /p "TARGET_PID=Inserisci l'ID del paziente [Default: %DEFAULT_PID%]: "
if "%TARGET_PID%"=="" set "TARGET_PID=%DEFAULT_PID%"

:DO_RUN
set "TARGET_PID=%TARGET_PID:"=%"
set "TARGET_PID=%TARGET_PID: =%"

echo.
echo ===============================================================================
if /i "%TARGET_PID%"=="SYNTHETIC" (
    echo Avvio inferenza con mammografia SINTETICA...
    echo ===============================================================================
    echo.
    "%PYTHON_CMD%" src\inference.py --synthetic --output "output\synthetic\breast_3d.glb"
    set "GENERATED_GLB=output\synthetic\breast_3d.glb"
) else if /i "%MODE%"=="UNIFIED" (
    echo Avvio RICOSTRUZIONE UNIFICATA per paziente: %TARGET_PID%
    echo (Superficie cutanea liscia Modulo 06 + Lesione interna 3D U-Net-GAN Modulo 07)
    echo ===============================================================================
    echo.
    "%PYTHON_CMD%" src\unified_reconstruction.py --patient_id "%TARGET_PID%"
    
    :: Cerca il file GLB unificato nella cartella output
    for /d %%D in (output\*!TARGET_PID!*) do (
        for %%F in (%%D\*unified*.glb) do (
            set "GENERATED_GLB=%%F"
        )
    )
) else (
    echo Avvio ricostruzione 3D U-Net-GAN per paziente: %TARGET_PID%
    echo ===============================================================================
    echo.
    "%PYTHON_CMD%" src\inference.py --patient_id "%TARGET_PID%"
    
    :: Cerca il file GLB generato nella cartella output del paziente
    for /d %%D in (output\*!TARGET_PID!*) do (
        for %%F in (%%D\*.glb) do (
            set "GENERATED_GLB=%%F"
        )
    )
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERRORE] Si e verificato un problema durante l'inferenza.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===============================================================================
echo  Ricostruzione 3D completata con successo!
if defined GENERATED_GLB echo  File GLB: %GENERATED_GLB%
echo ===============================================================================
echo.

:: Percorso Blender
set "BLENDER_EXE=C:\Users\Pierluca\OneDrive\Documenti\Blender\blender.exe"
if not exist "%BLENDER_EXE%" set "BLENDER_EXE=..\Blender\blender.exe"
if not exist "%BLENDER_EXE%" set "BLENDER_EXE=blender"

:: Genera automaticamente la scena .blend se Blender e presente
if defined GENERATED_GLB (
    set "GENERATED_BLEND=!GENERATED_GLB:.glb=.blend!"
    if exist "%BLENDER_EXE%" (
        echo Configurazione scena Blender con materiali e trasparenze...
        "%BLENDER_EXE%" --background --python src\blender_setup.py -- "%CD%\!GENERATED_GLB!" "%CD%\!GENERATED_BLEND!" >nul 2>&1
        if exist "!GENERATED_BLEND!" (
            echo  File Scena Blender: !GENERATED_BLEND!
        )
    )
)

echo.
echo ===============================================================================
echo  Vuoi aprire il modello generato in Blender adesso? (S/N)
echo ===============================================================================
set /p "OPEN_CHOICE=[S/N, Default: S]: "
if /i "%OPEN_CHOICE%"=="" set "OPEN_CHOICE=S"

if /i "%OPEN_CHOICE%"=="S" (
    if defined GENERATED_BLEND (
        if exist "!GENERATED_BLEND!" (
            echo Apertura scena in Blender: !GENERATED_BLEND!...
            start "" "!BLENDER_EXE!" "%CD%\!GENERATED_BLEND!"
            goto END
        )
    )
    if defined GENERATED_GLB (
        if exist "!GENERATED_GLB!" (
            echo Apertura modello in Blender: !GENERATED_GLB!...
            start "" "!BLENDER_EXE!" --python-expr "import bpy; bpy.ops.wm.read_homefile(use_empty=True); bpy.ops.import_scene.gltf(filepath=r'%CD%\!GENERATED_GLB!')"
            goto END
        )
    )
    echo File non trovato automaticamente. Controlla la cartella output.
)

:END
pause
