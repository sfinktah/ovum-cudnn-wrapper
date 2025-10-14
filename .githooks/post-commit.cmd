@echo off
REM Post-commit hook to auto-bump versions in pyproject.toml for ovum projects.
REM Requires Python available on PATH.

REM Prevent infinite recursion when we amend the commit
if "%GIT_BUMPING%"=="1" (
  goto :eof
)

setlocal ENABLEDELAYEDEXPANSION

REM Determine repository root
for /f "tokens=*" %%i in ('git rev-parse --show-toplevel') do set REPO_ROOT=%%i

REM List of per-project bump scripts (run individually)
set S1=%REPO_ROOT%\tools\auto_bump_version.py
set S2=%REPO_ROOT%\tools\auto_bump_version_spotlight.py
set S3=%REPO_ROOT%\tools\auto_bump_version_cudnn_wrapper.py

set CHANGED=

for %%S in ("%S1%" "%S2%" "%S3%") do (
  if exist %%~S (
    for /f "usebackq tokens=*" %%l in (`python %%~S`) do (
      set LINE=%%l
      if "!LINE!"=="CHANGED=1" set CHANGED=1
      for /f "tokens=1,2 delims==" %%a in ("!LINE!") do (
        if "%%a"=="FILE" set "FILEPATH=%%b" & git add "!FILEPATH!"
      )
    )
  )
)

if not "%CHANGED%"=="1" (
  endlocal
  goto :eof
)

REM Amend the just-created commit to include the bumped versions
set GIT_BUMPING=1
git commit --amend --no-edit >nul 2>&1
endlocal

exit /b 0
