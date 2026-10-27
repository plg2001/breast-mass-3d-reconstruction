@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Ricostruzione 3D Superficie Mammella da Mammografia

echo ========================================================
echo  Ricostruzione 3D Superficie Mammella (Skin Envelope)
echo  Viste Bi-Planari CC + MLO (JPEG / DICOM)
echo ========================================================
echo.

set DEFAULT_ID=P_00005
set PATIENT_ID=%~1

if "%PATIENT_ID%"=="" (
    echo Esempi di ID presenti in archivio: P_00001, P_00004, P_00005, P_00007, P_00008, P_00011...
    echo.
    set /p "PATIENT_ID=Inserisci l'ID del paziente [Default: !DEFAULT_ID!]: "
)

if "%PATIENT_ID%"=="" set PATIENT_ID=%DEFAULT_ID%

:: Rimuovi eventuali virgolette o spazi accidentali
set PATIENT_ID=%PATIENT_ID:"=%
set PATIENT_ID=%PATIENT_ID: =%

echo.
echo ========================================================
echo  Paziente selezionato: %PATIENT_ID%
echo ========================================================
echo.

:: Verifica esistenza cartella dati
if not exist "..\data\%PATIENT_ID%" (
    if not exist "data\%PATIENT_ID%" (
        echo [ERRORE] Cartella non trovata per il paziente: %PATIENT_ID%
        echo Verifica che la cartella 'data\%PATIENT_ID%' esista nel progetto.
        echo.
        pause
        exit /b 1
    )
)

echo Avvio ricostruzione 3D in corso...
python perfect_breast_reconstruction.py --patient_id %PATIENT_ID%

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo  Ricostruzione completata con successo!
    echo  Modello 3D salvato in: output\%PATIENT_ID%\
    echo ========================================================
    echo.
    set /p "OPEN_CHOICE=Vuoi aprire subito il modello in Blender? (S/N) [Default: S]: "
    if "!OPEN_CHOICE!"=="" set OPEN_CHOICE=S
    if /i "!OPEN_CHOICE!"=="S" (
        call open_in_blender.bat %PATIENT_ID%
    )
) else (
    echo.
    echo [ERRORE] Si e' verificato un problema durante la ricostruzione per il paziente %PATIENT_ID%.
    pause
)
