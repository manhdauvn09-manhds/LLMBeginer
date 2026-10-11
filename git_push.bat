@echo off
rem Commit toan bo thay doi tu local va day len GitHub.
cd /d "%~dp0"
echo ===============================
echo  Cac thay doi hien tai:
echo ===============================
git status --short
echo.
set /p msg="Nhap mo ta cho commit (Enter = dung mac dinh): "
if "%msg%"=="" set msg=Cap nhat code
git add -A
git commit -m "%msg%"
if errorlevel 1 (
    echo.
    echo Khong co gi de commit, hoac co loi.
    pause
    exit /b 1
)
echo.
echo Day len GitHub...
git push
if errorlevel 1 (
    echo.
    echo LOI khi push! Thu chay git_pull.bat truoc roi push lai.
    pause
    exit /b 1
)
echo.
echo Xong! Da day len GitHub.
pause
