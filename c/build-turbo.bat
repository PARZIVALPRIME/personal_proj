@echo off
rem ---------------------------------------------------------------------------
rem build-turbo.bat - qwen36.exe for THIS machine (Ryzen 7 5800HS, AVX2) with
rem LLVM-MinGW. No nvcc or MSVC needed for either variant.
rem
rem   build-turbo.bat          CPU engine (the default, measured fastest so far)
rem   build-turbo.bat cuda     + CUDA tier through coli_cuda.dll (RTX 3050). The
rem                            DLL is copied from the installed 2.1.0 runtime, the
rem                            same version as this source tree.
rem   build-turbo.bat vk       + Vulkan trunk chain and expert tier (RTX 3050 through
rem                            the NVIDIA Vulkan driver). Needs the Vulkan SDK
rem                            headers (winget install KhronosGroup.VulkanSDK); the
rem                            SPIR-V shaders are already in c\shaders.
rem
rem Overrides (set before calling): LLVM_MINGW (toolchain root), COLI_RUNTIME
rem (installed runtime holding coli_cuda.dll and the CUDA DLLs).
rem ---------------------------------------------------------------------------
setlocal
if "%LLVM_MINGW%"=="" set "LLVM_MINGW=C:\Users\riyaz\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20261020-ucrt-x86_64"
if "%COLI_RUNTIME%"=="" set "COLI_RUNTIME=C:\Users\riyaz\AppData\Local\colibri\runtime\colibri-v2.1.0"
if not exist "%LLVM_MINGW%\bin\clang.exe" (
    echo ERROR: LLVM-MinGW not found at %LLVM_MINGW% - set LLVM_MINGW first.>&2
    exit /b 1
)
set "PATH=%LLVM_MINGW%\bin;%PATH%"
set "OMPLIB=%LLVM_MINGW%\x86_64-w64-mingw32\lib"
set "CFL=-D_FILE_OFFSET_BITS=64 -O3 -march=native -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function"
set "LDF=-lm -fopenmp -pthread -lpsapi -L%OMPLIB%"
pushd "%~dp0"

if exist qwen36.exe if not exist qwen36-before-turbo.exe copy /y qwen36.exe qwen36-before-turbo.exe >nul

if /i "%~1"=="cuda" goto cuda
if /i "%~1"=="vk" goto vk

echo Building qwen36.exe (CPU, -march=native) ...
clang %CFL% qwen36.c -o qwen36.exe %LDF%
if errorlevel 1 goto failed
goto done

:vk
if "%VULKAN_SDK%"=="" (
    echo ERROR: VULKAN_SDK is not set. Install the Vulkan SDK, open a new prompt, run again.>&2
    popd
    exit /b 1
)
set "VKF=-DCOLI_VULKAN -I"%VULKAN_SDK%\Include""
echo Building qwen36.exe (CPU + Vulkan chain and tier) ...
clang %CFL% %VKF% -c backend_vulkan.c -o backend_vulkan.o || goto failed
clang %CFL% %VKF% -c vk_tier.c -o vk_tier.o || goto failed
clang %CFL% %VKF% -c vk_chain.c -o vk_chain.o || goto failed
clang %CFL% %VKF% qwen36.c backend_vulkan.o vk_tier.o vk_chain.o -o qwen36.exe %LDF%
if errorlevel 1 goto failed
goto done

:cuda
echo Building qwen36.exe (CPU + CUDA tier via coli_cuda.dll) ...
clang %CFL% -DCOLI_CUDA -c qwen36_tier.c -o qwen36_tier.o || goto failed
clang %CFL% -DCOLI_CUDA -c backend_loader.c -o backend_loader.o || goto failed
clang %CFL% -DCOLI_CUDA -c cu_tier.c -o cu_tier.o || goto failed
clang %CFL% -DCOLI_CUDA qwen36.c qwen36_tier.o backend_loader.o cu_tier.o -o qwen36.exe %LDF%
if errorlevel 1 goto failed
if exist "%COLI_RUNTIME%\coli_cuda.dll" (
    copy /y "%COLI_RUNTIME%\coli_cuda.dll" coli_cuda.dll >nul
    echo Copied coli_cuda.dll from %COLI_RUNTIME%
) else (
    echo WARNING: %COLI_RUNTIME%\coli_cuda.dll not found: the engine will run CPU-only.>&2
    echo          Build it with build_cuda.bat ^(needs nvcc + MSVC^) or point COLI_RUNTIME at a runtime that has it.>&2
)

:done
popd
echo.
echo OK: %~dp0qwen36.exe
echo Previous engine kept as qwen36-before-turbo.exe for A/B runs.
echo Run it with ..\benchmark-results\turbo-qwen36.bat
exit /b 0

:failed
popd
echo.>&2
echo BUILD FAILED. If make worked before, the old command still works too:>&2
echo   make -C c qwen36 ARCH=native LDFLAGS="-lm -fopenmp -pthread -lpsapi -L%OMPLIB%">&2
exit /b 1
