@echo off
setlocal
set "SNAP=D:\colibri-models\qwen36-35b"
set "TOK=%SNAP%\tokenizer.json"
set "COLI_CUDA=0"
set "COLI_VULKAN=0"
set "DIRECT=0"
set "PIPE=0"
set "PILOT_REAL=0"
set "COLI_LOOKUP=1"
set "OMP_NUM_THREADS=8"
set "Q36_MAXT=2048"
set "COLI_ENGINE=D:\colibri-main\colibri-main\c\qwen36.exe"
set "LLVM_MINGW_BIN=C:\Users\riyaz\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20261020-ucrt-x86_64\bin"
set "PATH=%LLVM_MINGW_BIN%;%PATH%"
call "C:\Users\riyaz\AppData\Local\colibri\runtime\colibri-v2.1.0\coli.cmd" serve --model "%SNAP%" --cap 256 %*
exit /b %ERRORLEVEL%
