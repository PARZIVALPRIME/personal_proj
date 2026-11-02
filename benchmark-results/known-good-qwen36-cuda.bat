@echo off
setlocal
set "SNAP=D:\colibri-models\qwen36-35b"
set "TOK=D:\colibri-models\qwen36-35b\tokenizer.json"
set "COLI_CUDA=1"
set "COLI_GPUS=0"
set "CUDA_EXPERT_GB=1"
set "COLI_VULKAN=0"
set "OMP_NUM_THREADS=8"
set "Q36_MAXT=2048"
set "COLI_MODEL=%SNAP%"
call "C:\Users\riyaz\AppData\Local\colibri\runtime\colibri-v2.1.0\coli.cmd" serve --model "%SNAP%" %*
exit /b %ERRORLEVEL%
