@echo off
title Ricostruzione 3D Lesione Mammaria SMRD (De Sisternes et al. 2015)
echo =========================================================================
echo   SMRD 3D Lesion Reconstruction Pipeline (De Sisternes et al. 2015)
echo =========================================================================
echo.

set PYTHON_EXE=C:\Python314\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

echo [1/2] Esecuzione Ricostruzione SMRD per tutti i pazienti...
"%PYTHON_EXE%" "%~dp0smrd_lesion_reconstruction.py" --patient all
if %ERRORLEVEL% neq 0 (
    echo [ERRORE] Esecuzione fallita!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/2] Generazione Confronto Diretto vs Ling Shao (2007)...
"%PYTHON_EXE%" "%~dp0compare_ling_shao_vs_smrd.py"

echo.
echo =========================================================================
echo   COMPLETATO CON SUCCESSO!
echo   I file 3D (.obj, .glb), i render e i report sono in:
echo   %~dp0output\
echo =========================================================================
echo.
pause
