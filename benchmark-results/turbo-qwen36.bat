@echo off
rem ---------------------------------------------------------------------------
rem turbo-qwen36.bat - Qwen3.6-35B-A3B (i4-gs64) on Ryzen 7 5800HS + RTX 3050 4 GB
rem                    + 16 GB RAM, with the engine built by c\build-turbo.bat.
rem
rem   turbo-qwen36.bat              OpenAI-compatible server (coli serve)
rem   turbo-qwen36.bat chat         interactive chat
rem   turbo-qwen36.bat bench        direct engine run on prompt.txt, timers on,
rem                                 writes bench-turbo-<n>.log (run it 3x; run 1
rem                                 also builds the heat history that 2 and 3 use)
rem
rem Switches (edit below or set before calling):
rem   GPU=vk    trunk chain + hot experts on the RTX 3050 through Vulkan; the trunk's
rem             host copy is dropped and that RAM goes to the expert cache
rem             (build-turbo.bat vk). Prompt lookup stays on.
rem   GPU=1     dense trunk on the RTX 3050 through CUDA (build-turbo.bat cuda)
rem   LOSSY=1   skip low-weight experts that are not in RAM (Q36_SKIP_MISS)
rem   PILOT=1   cross-layer expert prefetch (now event-driven; A/B it)
rem ---------------------------------------------------------------------------
setlocal
if "%SNAP%"=="" set "SNAP=D:\colibri-models\qwen36-35b"
if "%SRC%"=="" set "SRC=D:\colibri-main\colibri-main"
if "%COLI_RUNTIME%"=="" set "COLI_RUNTIME=C:\Users\riyaz\AppData\Local\colibri\runtime\colibri-v2.1.0"
if "%LLVM_MINGW_BIN%"=="" set "LLVM_MINGW_BIN=C:\Users\riyaz\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20261020-ucrt-x86_64\bin"
if "%GPU%"=="" set "GPU=0"
if "%LOSSY%"=="" set "LOSSY=0"
if "%PILOT%"=="" set "PILOT=0"

set "TOK=%SNAP%\tokenizer.json"
set "COLI_ENGINE=%SRC%\c\qwen36.exe"
rem libomp.dll for the LLVM-MinGW build; the runtime dir for coli_cuda.dll's CUDA DLLs
set "PATH=%SRC%\c;%LLVM_MINGW_BIN%;%COLI_RUNTIME%;%PATH%"

rem --- CPU: 8 physical cores; SMT siblings are left to the expert loaders -------
set "OMP_NUM_THREADS=8"
rem --- SSD: serialized reads are the stable default for this safetensors build.
rem     Set Q36_IO_THREADS=8 explicitly only after validating the loader path.
if "%Q36_IO_THREADS%"=="" set "Q36_IO_THREADS=0"
rem --- RAM cache: sized from measured free RAM at the real slot size (cap 0) -----
rem     segmented LRU keeps the experts routed again and again; the heat history
rem     refills the cache at every start instead of starting cold
set "Q36_EVICT=slru"
set "Q36_SLRU_PROT=0.8"
set "Q36_HEAT=1"
rem --- context: 16k positions for agent/tool prompts (20 MB of KV per 1k) --------
if "%Q36_MAXT%"=="" set "Q36_MAXT=16384"
rem --- speculative prompt lookup: exact, big win on code edits ------------------
set "COLI_LOOKUP=1"
set "COLI_VULKAN=0"

if "%PILOT%"=="1" (
    set "PILOT_WORKERS=4"
) else (
    set "PILOT=0"
)

if "%LOSSY%"=="1" (
    rem rank 5..8 experts under 6%% of the router mass are skipped when not in RAM
    set "Q36_SKIP_MISS=0.06"
    set "Q36_SKIP_KEEP=4"
)

if /i "%GPU%"=="vk" (
    set "COLI_CUDA=0"
    set "COLI_VULKAN=1"
    set "COLI_VK_SHADERS=%SRC%\c\shaders"
) else if "%GPU%"=="1" (
    set "COLI_CUDA=1"
    set "COLI_GPUS=0"
    rem the whole dense trunk on the card, every routed expert on the CPU/RAM/SSD path
    set "COLI_PLACE=lmhead=0,dnproj=0,dnout=0,attnproj=0,shexp=0,experts=cpu"
    set "COLI_CUDA_CHAIN=1"
    rem keep prompt-lookup drafts beside a trunk-only CUDA tier
    set "Q36_SPEC_LOOSE=1"
) else (
    set "COLI_CUDA=0"
)

if not exist "%COLI_ENGINE%" (
    echo ERROR: %COLI_ENGINE% not found. Run %SRC%\c\build-turbo.bat first.>&2
    exit /b 1
)

if /i "%~1"=="bench" goto bench
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
set "COLI_TEMP=0"
set "Q36_HEAT_WAIT=1"
set /a "BN=1"
:findlog
if exist "%~dp0bench-turbo-%BN%.log" ( set /a "BN+=1" & goto findlog )
echo Writing %~dp0bench-turbo-%BN%.log ...
"%COLI_ENGINE%" 0 4 "%~dp0prompt.txt" > "%~dp0bench-turbo-%BN%.log" 2>&1
set "RC=%ERRORLEVEL%"
findstr /c:"cache auto-sized" /c:"loader threads" /c:"Q36_HEAT" /c:"tok/s" /c:"hit rate" /c:"[timers]" "%~dp0bench-turbo-%BN%.log"
exit /b %RC%
