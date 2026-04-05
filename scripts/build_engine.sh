#!/usr/bin/env bash
# ============================================
# MIND EASE — Build C++ Inference Engine
# ============================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$ROOT/engine"
BUILD="$ENGINE/build"

echo "Building MIND EASE engine..."

# Check prerequisites
command -v cmake  >/dev/null 2>&1 || { echo "cmake not found. Install via: brew install cmake"; exit 1; }
command -v git    >/dev/null 2>&1 || { echo "git not found"; exit 1; }

# Init submodules
cd "$ENGINE"
if [ ! -f "third_party/whisper.cpp/CMakeLists.txt" ]; then
    echo "Fetching whisper.cpp..."
    git clone --depth 1 https://github.com/ggerganov/whisper.cpp third_party/whisper.cpp
fi
if [ ! -f "third_party/llama.cpp/CMakeLists.txt" ]; then
    echo "Fetching llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp third_party/llama.cpp
fi
if [ ! -d "third_party/onnxruntime/include" ]; then
    echo "Downloading ONNX Runtime..."
    ONNX_VERSION="1.19.2"
    if [[ "$(uname)" == "Darwin" ]]; then
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-osx-universal2-${ONNX_VERSION}.tgz"
        curl -L "$ONNX_URL" | tar -xz -C third_party/
        mv "third_party/onnxruntime-osx-universal2-${ONNX_VERSION}" third_party/onnxruntime
    else
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
        curl -L "$ONNX_URL" | tar -xz -C third_party/
        mv "third_party/onnxruntime-linux-x64-${ONNX_VERSION}" third_party/onnxruntime
    fi
fi

# Configure
mkdir -p "$BUILD"
cd "$BUILD"

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
if [[ "$(uname)" == "Darwin" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DMINDEASE_USE_METAL=ON"
fi

cmake .. $CMAKE_ARGS
cmake --build . -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

echo "Engine built: $BUILD/mindease_engine"
echo "Running self-test..."
"$BUILD/mindease_engine" --test
