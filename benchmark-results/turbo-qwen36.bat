@echo off
rem ---------------------------------------------------------------------------
rem turbo-qwen36.bat - Qwen3.6-35B-A3B (i4-gs64) on Ryzen 7 5800HS + RTX 3050 4 GB
rem                    + 16 GB RAM, with the engine built by c\build-turbo.bat.
rem
rem   turbo-qwen36.bat              OpenAI-compatible server (coli serve)
rem   turbo-qwen36.bat chat         interactive chat
rem   turbo-qwen36.bat bench        one direct run on prompt.txt -> bench-turbo-<n>.log
rem   turbo-qwen36.bat sweep        every config below, one run each, warm, with a
rem                                 summary in sweep-<n>\summary.txt (build with
rem                                 "build-turbo.bat vk" first: that binary runs both
rem                                 the CPU and the Vulkan configs)
rem
rem Switches (set before calling):
rem   GPU=vk    trunk chain + hot experts on the RTX 3050 through Vulkan; the trunk's
rem             host copy is dropped and that RAM goes to the expert cache
rem   GPU=1     dense trunk on the RTX 3050 through CUDA (build-turbo.bat cuda)
rem   LOSSY=1   skip low-weight experts not in RAM + lm_head in int4 (~+1.5% ppl)
rem   LOSSY=2   LOSSY=1 + DeltaNet projections in int4 (~+5.6% ppl, ~0.75 GB/token less)
rem   PILOT=1   cross-layer expert prefetch (CPU trunk only; the GPU chains refuse it)
rem ---------------------------------------------------------------------------
setlocal
set "HERE=%~dp0"
if "%SNAP%"=="" set "SNAP=D:\colibri-models\qwen36-35b"
if "%SRC%"=="" set "SRC=D:\colibri-main\colibri-main"
if "%COLI_RUNTIME%"=="" set "COLI_RUNTIME=C:\Users\riyaz\AppData\Local\colibri\runtime\colibri-v2.1.0"
if "%LLVM_MINGW_BIN%"=="" set "LLVM_MINGW_BIN=C:\Users\riyaz\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20261020-ucrt-x86_64\bin"
if "%GPU%"=="" set "GPU=0"
if "%LOSSY%"=="" set "LOSSY=0"
if "%PILOT%"=="" set "PILOT=0"

set "TOK=%SNAP%\tokenizer.json"
set "COLI_ENGINE=%SRC%\c\qwen36.exe"
set "COLI_VK_SHADERS=%SRC%\c\shaders"
rem libomp.dll for the LLVM-MinGW build; the runtime dir for coli_cuda.dll's CUDA DLLs
set "PATH=%SRC%\c;%LLVM_MINGW_BIN%;%COLI_RUNTIME%;%PATH%"

rem --- CPU: 8 physical cores; SMT siblings are left to the expert loaders -------
set "OMP_NUM_THREADS=8"
rem --- SSD: a layer's misses read 8 at a time while its residents compute -------
if "%Q36_IO_THREADS%"=="" set "Q36_IO_THREADS=8"
rem --- RAM cache: sized from free RAM at the real slot size (cap 0), segmented ---
rem     LRU, refilled from the heat history at every start
set "Q36_EVICT=slru"
set "Q36_SLRU_PROT=0.8"
set "Q36_HEAT=1"
rem --- context: 16k positions for agent/tool prompts (20 MB of KV per 1k) --------
if "%Q36_MAXT%"=="" set "Q36_MAXT=16384"
rem --- speculative prompt lookup: exact, big win on code edits ------------------
set "COLI_LOOKUP=1"
set "COLI_VULKAN=0"
set "COLI_CUDA=0"

if "%PILOT%"=="1" (
    set "PILOT_WORKERS=4"
) else (
    set "PILOT=0"
)
if "%LOSSY%"=="1" call :lossy1
if "%LOSSY%"=="2" call :lossy2
if /i "%GPU%"=="vk" call :gpu_vk
if "%GPU%"=="1" call :gpu_cuda

if not exist "%COLI_ENGINE%" (
    echo ERROR: %COLI_ENGINE% not found. Run %SRC%\c\build-turbo.bat first.>&2
    exit /b 1
)

if /i "%~1"=="bench" goto bench
if /i "%~1"=="sweep" goto sweep
if /i "%~1"=="chat" goto chat

call "%COLI_RUNTIME%\coli.cmd" serve --model "%SNAP%" --cap 0 %*
exit /b %ERRORLEVEL%

:chat
shift
call "%COLI_RUNTIME%\coli.cmd" chat --model "%SNAP%" --cap 0 %1 %2 %3 %4 %5 %6
exit /b %ERRORLEVEL%

:bench
if "%N_NEW%"=="" set "N_NEW=128"
set "COLI_TIMERS=1"
set "Q36_HEAT_WAIT=1"
set /a "BN=1"
:findlog
if exist "%HERE%bench-turbo-%BN%.log" ( set /a "BN+=1" & goto findlog )
echo Writing %HERE%bench-turbo-%BN%.log ...
"%COLI_ENGINE%" 0 4 "%HERE%prompt.txt" > "%HERE%bench-turbo-%BN%.log" 2>&1
set "RC=%ERRORLEVEL%"
findstr /c:"cache auto-sized" /c:"loader threads" /c:"Q36_HEAT" /c:"Speed:" /c:"hit rate" /c:"[timers]" /c:"[VK]" "%HERE%bench-turbo-%BN%.log"
exit /b %RC%

:sweep
if "%N_NEW%"=="" set "N_NEW=128"
set "COLI_TIMERS=1"
set "Q36_HEAT_WAIT=1"
set /a "SN=1"
:findsweep
if exist "%HERE%sweep-%SN%" ( set /a "SN+=1" & goto findsweep )
set "SWEEP=%HERE%sweep-%SN%"
mkdir "%SWEEP%"
echo Sweep logs in %SWEEP% (N_NEW=%N_NEW%)
call :one cpu-io0            "Q36_IO_THREADS=0"
call :one cpu-io8            "Q36_IO_THREADS=8"
call :one cpu-io8-pilot      "Q36_IO_THREADS=8" "PILOT=1" "PILOT_WORKERS=4"
call :one cpu-io8-ram92      "Q36_IO_THREADS=8" "Q36_RAM_FRAC=0.92"
call :one vk                 "Q36_IO_THREADS=8" "COLI_VULKAN=1"
call :one vk-notier          "Q36_IO_THREADS=8" "COLI_VULKAN=1" "COLI_VK_TIER=0"
call :one cpu-lossy1         "Q36_IO_THREADS=8" "Q36_SKIP_MISS=0.06" "COLI_DENSE_BITS=4" "COLI_DENSE_INT4=lmhead"
call :one vk-lossy1          "Q36_IO_THREADS=8" "COLI_VULKAN=1" "Q36_SKIP_MISS=0.06"
call :one cpu-lossy2         "Q36_IO_THREADS=8" "Q36_SKIP_MISS=0.08" "COLI_DENSE_BITS=4" "COLI_DENSE_INT4=lmhead,dnproj,dnout"
echo.
echo ===== %SWEEP%\summary.txt =====
type "%SWEEP%\summary.txt"
exit /b 0

rem one sweep config: name, then "VAR=value" pairs
:one
setlocal
set "NAME=%~1"
:one_arg
shift
if "%~1"=="" goto one_go
set "%~1"
goto one_arg
:one_go
echo [%NAME%] running ...
"%COLI_ENGINE%" 0 4 "%HERE%prompt.txt" > "%SWEEP%\%NAME%.log" 2>&1
set "RC=%ERRORLEVEL%"
>> "%SWEEP%\summary.txt" echo ---- %NAME% (exit %RC%)
findstr /c:"Speed:" /c:"hit rate" /c:"expert kernel" /c:"TOTAL" /c:"cache auto-sized" /c:"expert cache" /c:"dense chain" /c:"Q36_SKIP_MISS:" "%SWEEP%\%NAME%.log" >> "%SWEEP%\summary.txt"
endlocal
goto :eof

:lossy1
set "Q36_SKIP_MISS=0.06"
set "Q36_SKIP_KEEP=4"
set "COLI_DENSE_BITS=4"
set "COLI_DENSE_INT4=lmhead"
goto :eof

:lossy2
set "Q36_SKIP_MISS=0.08"
set "Q36_SKIP_KEEP=4"
set "COLI_DENSE_BITS=4"
set "COLI_DENSE_INT4=lmhead,dnproj,dnout"
goto :eof

:gpu_vk
set "COLI_VULKAN=1"
set "PILOT=0"
goto :eof

:gpu_cuda
set "COLI_CUDA=1"
set "COLI_GPUS=0"
rem the whole dense trunk on the card, every routed expert on the CPU/RAM/SSD path
set "COLI_PLACE=lmhead=0,dnproj=0,dnout=0,attnproj=0,shexp=0,experts=cpu"
set "COLI_CUDA_CHAIN=1"
set "Q36_SPEC_LOOSE=1"
set "PILOT=0"
goto :eof
