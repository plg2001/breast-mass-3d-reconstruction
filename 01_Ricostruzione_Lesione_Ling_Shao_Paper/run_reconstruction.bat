@echo off
title Ricostruzione 3D Lesione Mammografica (Ling Shao, 2007 - PVC)
echo ===============================================================================
echo   RICOSTRUZIONE 3D LESIONE DA MAMMOGRAFIA (LING SHAO, 2007)
echo   Gestione Multi-Paziente (Output separati per cartella)
echo ===============================================================================
echo.
cd /d "%~dp0"

echo Scegli quale caso clinico ricostruire:
echo   [1] Ricostruisci TUTTI i pazienti trovati in data/ (--all-cases)
echo   [2] Solo paziente P_00232 (output in output\P_00232\)
echo   [3] Solo paziente P_01754 (output in output\P_01754\)
echo.
set /p SCELTA="Inserisci opzione (1, 2 o 3, default: 1): "

if "%SCELTA%"=="2" (
    echo.
    echo Esecuzione ricostruzione per P_00232...
    python lesion_pvc_reconstruction.py --case P_00232
) else if "%SCELTA%"=="3" (
    echo.
    echo Esecuzione ricostruzione per P_01754...
    python lesion_pvc_reconstruction.py --case P_01754
) else (
    echo.
    echo Esecuzione batch per TUTTI i pazienti in data\...
    python lesion_pvc_reconstruction.py --all-cases
)

echo.
echo ===============================================================================
echo Elaborazione completata! Risultati organizzati in output/[Paziente]/
echo ===============================================================================
pause
