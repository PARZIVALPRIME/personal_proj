@echo off
setlocal
set "SNAP=D:\colibri-models\qwen36-35b"
set "TOK=%SNAP%\tokenizer.json"
set "COLI_CUDA=1"
set "COLI_GPUS=0"
set "CUDA_EXPERT_GB=1.5"
set "COLI_PLACE=experts=0,lmhead=0,dnproj=cpu,dnout=cpu,attnproj=cpu"
set "COLI_CUDA_CHAIN=0"
set "COLI_VULKAN=0"
set "COLI_LOOKUP=0"
set "OMP_NUM_THREADS=8"
set "Q36_MAXT=2048"
call "C:\Users\riyaz\AppData\Local\colibri\runtime\colibri-v2.1.0\coli.cmd" serve --model "%SNAP%" --cap 64 %*
exit /b %ERRORLEVEL%
