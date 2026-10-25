@echo off
setlocal
cd /d "%~dp0"
title Ricostruzione 3D Superficie Mammella da Mammografia

echo ========================================================
echo  Ricostruzione 3D Superficie Mammella (Skin Envelope)
echo  Viste Bi-Planari CC + MLO (JPEG / DICOM)
echo ========================================================
echo.

set PATIENT_ID=P_00005
if not "%~1"=="" set PATIENT_ID=%~1

echo Elaborazione Paziente: %PATIENT_ID% ...
python breast_surface_reconstruction.py --patient_id %PATIENT_ID%

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo  Ricostruzione completata con successo!
    echo  Output salvato in: output\%PATIENT_ID%\
    echo ========================================================
    if exist "output\%PATIENT_ID%\%PATIENT_ID%_breast_surface_multiview.png" (
        start "" "output\%PATIENT_ID%\%PATIENT_ID%_breast_surface_multiview.png"
    )
) else (
    echo.
    echo [ERRORE] Si e' verificato un problema durante la ricostruzione.
)

pause
