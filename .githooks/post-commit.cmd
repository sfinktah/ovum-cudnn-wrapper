@echo off
REM Generic post-commit hook to auto-bump a version in a single target file.
REM Requires Python available on PATH.

REM Prevent infinite recursion when we amend the commit
if "%GIT_BUMPING%"=="1" (
  goto :eof
)

setlocal ENABLEDELAYEDEXPANSION

REM Resolve repository root (folder containing this .githooks directory)
set "SCRIPT_DIR=%~dp0"
for %%# in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~f#"

REM Allow overrides via environment; provide sensible defaults
if not defined BUMP_SCRIPT set "BUMP_SCRIPT=%REPO_ROOT%\tools\auto_bump_version.py"
if not defined BUMP_TARGET_FILE set "BUMP_TARGET_FILE=%REPO_ROOT%\pyproject.toml"

if exist "%BUMP_SCRIPT%" (
  pushd "%REPO_ROOT%" >nul 2>&1
  REM Run the bump script; it returns 1 if a change was made
  python "%BUMP_SCRIPT%" --file "%BUMP_TARGET_FILE%" >nul 2>&1
  if errorlevel 1 (
    git add "%BUMP_TARGET_FILE%"
    REM Amend the just-created commit to include the bumped version
    set GIT_BUMPING=1
    git commit --amend --no-edit >nul 2>&1
  )
  popd >nul 2>&1
)

endlocal
exit /b 0
