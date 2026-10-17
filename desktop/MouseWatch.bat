@echo off
cd /d "%~dp0"
where pythonw >nul 2>&1 && start "" pythonw -m mousewatch && exit /b 0
where pyw >nul 2>&1 && start "" pyw -m mousewatch && exit /b 0
echo Python was not found on PATH.
exit /b 1
