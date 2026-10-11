@echo off
rem Lay toan bo code moi nhat tu GitHub ve may local.
cd /d "%~dp0"
echo ===============================
echo  Lay code moi nhat tu GitHub...
echo ===============================
git pull
if errorlevel 1 (
    echo.
    echo LOI khi pull! Kiem tra ket noi mang hoac xung dot file.
    pause
    exit /b 1
)
echo.
echo Xong! Da cap nhat code moi nhat.
pause
