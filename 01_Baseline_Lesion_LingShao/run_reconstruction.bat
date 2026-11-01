@echo off
title Ricostruzione 3D Lesione Mammografica (Ling Shao, 2007 - PVC)
echo ===============================================================================
echo   RICOSTRUZIONE 3D LESIONE DA MAMMOGRAFIA (LING SHAO, 2007)
echo   Gestione Multi-Paziente (Output separati per cartella)
echo ===============================================================================
echo.
cd /d "%~dp0"

set PYTHON_CMD=python
if exist "C:\Python314\python.exe" set PYTHON_CMD=C:\Python314\python.exe
if exist "C:\Python311\python.exe" set PYTHON_CMD=C:\Python311\python.exe

echo Scegli quale caso clinico ricostruire da data/:
echo   [1] Ricostruisci TUTTI i pazienti trovati in data/ (--all-cases)
echo   [2] Paziente P_00232 (output in output\P_00232\)
echo   [3] Paziente P_01754 (output in output\P_01754\)
echo   [4] Inserisci ID paziente personalizzato (es. P_00001, P_00506...)
echo.
set /p SCELTA="Inserisci opzione (1, 2, 3 o 4, default: 2): "

if "%SCELTA%"=="1" goto opt_all
if "%SCELTA%"=="3" goto opt_p01754
if "%SCELTA%"=="4" goto opt_custom
goto opt_p00232

:opt_all
echo.
echo Esecuzione batch per TUTTI i pazienti in data\...
"%PYTHON_CMD%" lesion_pvc_reconstruction.py --all-cases
goto fine

:opt_p01754
echo.
echo Esecuzione ricostruzione per P_01754...
"%PYTHON_CMD%" lesion_pvc_reconstruction.py --case P_01754
goto fine

:opt_custom
echo.
set /p CUSTOM_PATIENT="Inserisci il codice paziente (es. P_00001): "
echo Esecuzione ricostruzione per %CUSTOM_PATIENT%...
"%PYTHON_CMD%" lesion_pvc_reconstruction.py --case %CUSTOM_PATIENT%
goto fine

:opt_p00232
echo.
echo Esecuzione ricostruzione per P_00232...
"%PYTHON_CMD%" lesion_pvc_reconstruction.py --case P_00232
goto fine

:fine
echo.
echo ===============================================================================
echo Elaborazione completata! Risultati organizzati in output/[Paziente]/
echo ===============================================================================
pause
