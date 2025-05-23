# DSP Visualization Project

This project is a simple digital signal processing (DSP) visualization tool using **SFML**, **ImGui-SFML**, and **ImPlot**. It demonstrates signal generation, sampling, FFT, FIR filtering, quantization, and signal reconstruction with interactive plots.

## Features

- Generates and visualizes original and sampled signals
- Computes and displays FFT of signals
- Applies FIR filters (2kHz and 1.5kHz)
- Demonstrates quantization and signal reconstruction
- Interactive GUI with ImGui and ImPlot

## Requirements

- C++17 compatible compiler
- [SFML](https://www.sfml-dev.org/) (Simple and Fast Multimedia Library)
- [ImGui-SFML](https://github.com/eliasdaler/imgui-sfml)
- [ImPlot](https://github.com/epezent/implot)

> **Tip:** You can use [vcpkg](https://github.com/microsoft/vcpkg) to install dependencies easily:
> ```
> vcpkg install sfml imgui-sfml implot
> ```

## Build Instructions

1. **Clone the repository:**
    ```sh
    git clone https://github.com/hasankemal1/dsp.git
    cd dsp
    ```

2. **Open the project in Visual Studio:**
    - Open `dsp.sln` in Visual Studio 2022 or later.

3. **Configure dependencies:**
    - Make sure SFML, ImGui-SFML, and ImPlot are installed and included in your project settings.
    - If using vcpkg, integrate it with Visual Studio (`vcpkg integrate install`).

4. **Build the project:**
    - Select `Release` or `Debug` configuration.
    - Build the solution (`Ctrl+Shift+B`).

5. **Run:**
    - The executable will be in `x64/Release/` or `x64/Debug/` folder.
    - Make sure required DLLs (SFML, ImGui-SFML, ImPlot) are in the same folder as the executable.

## File Structure
"# dsp" 
#   d s p 
 
 
