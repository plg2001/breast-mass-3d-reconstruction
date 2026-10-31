@echo off
title Configurazione Ambiente Python - Mammography 3D
cd /d "%~dp0"

echo =====================================================================
echo  MAMMOGRAPHY 3D - CONFIGURAZIONE AUTOMATICA AMBIENTE PYTHON
echo =====================================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERRORE] Python non e installato o non e presente nel PATH di sistema.
    echo Per favore installa Python 3.10 o 3.11 da https://www.python.org/
    echo e assicurati di spuntare "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

echo [1/3] Creazione ambiente virtuale locale (.venv)...
python -m venv .venv
if %ERRORLEVEL% neq 0 (
    echo [ERRORE] Creazione dell'ambiente virtuale fallita.
    pause
    exit /b 1
)

echo [2/3] Aggiornamento pip...
call .venv\Scripts\python.exe -m pip install --upgrade pip

echo [3/3] Installazione librerie scientifiche e AI (PyTorch, SciPy, Trimesh, OpenCV)...
call .venv\Scripts\pip.exe install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [AVVISO] Installazione completata con avvisi. Verificare i log sopra.
) else (
    echo.
    echo =====================================================================
    echo  CONFIGURAZIONE COMPLETATA CON SUCCESSO!
    echo =====================================================================
    echo L'applicazione WPF rilevera automaticamente l'ambiente virtuale .venv.
    echo Puoi ora avviare la suite facendo doppio clic su 'Mammography3D.bat'.
    echo =====================================================================
)

echo.
pause
