@echo off

echo %Running kickerstarter DASH app

set "python=C:\ProgramData\Anaconda3\python.exe"
set "app=C:\Users\yeoshuiming\Dropbox\GitHub\py-learn\learn-dash\kickerstarter\app.py" REM DASH app

echo %python%
echo %app%

%python% -u %app%

echo %ERRORLEVEL% :: handling error 1=error present 0=no error
echo %End of File
pause rem call cmd