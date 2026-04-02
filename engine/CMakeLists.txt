cmake_minimum_required(VERSION 3.20)
project(mindease_engine VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================
# Build options
# ============================================
option(MINDEASE_USE_METAL   "Enable Apple Metal acceleration"  OFF)
option(MINDEASE_USE_CUDA    "Enable NVIDIA CUDA acceleration"  OFF)
option(MINDEASE_BUILD_TESTS "Build unit tests"                 OFF)

# Auto-detect Metal on macOS
if(APPLE AND NOT MINDEASE_USE_CUDA)
    set(MINDEASE_USE_METAL ON)
    message(STATUS "Detected macOS — enabling Metal acceleration")
endif()

# ============================================
# Third-party dependencies
# ============================================

# --- whisper.cpp ---
set(WHISPER_DIR ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp)
if(EXISTS ${WHISPER_DIR}/CMakeLists.txt)
    set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    if(MINDEASE_USE_METAL)
        set(WHISPER_METAL ON CACHE BOOL "" FORCE)
    endif()
    if(MINDEASE_USE_CUDA)
        set(WHISPER_CUDA ON CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(${WHISPER_DIR} whisper_build EXCLUDE_FROM_ALL)
    set(WHISPER_AVAILABLE ON)
    message(STATUS "whisper.cpp found and configured")
else()
    set(WHISPER_AVAILABLE OFF)
    message(STATUS "whisper.cpp not found — STT will use stub")
endif()

# --- llama.cpp ---
set(LLAMA_DIR ${CMAKE_SOURCE_DIR}/third_party/llama.cpp)
if(EXISTS ${LLAMA_DIR}/CMakeLists.txt)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    if(MINDEASE_USE_METAL)
        set(GGML_METAL ON CACHE BOOL "" FORCE)
    endif()
    if(MINDEASE_USE_CUDA)
        set(GGML_CUDA ON CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(${LLAMA_DIR} llama_build EXCLUDE_FROM_ALL)
    set(LLAMA_AVAILABLE ON)
    message(STATUS "llama.cpp found and configured")
else()
    set(LLAMA_AVAILABLE OFF)
    message(STATUS "llama.cpp not found — LLM will use stub")
endif()

# --- ONNX Runtime ---
set(ONNX_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
if(EXISTS ${ONNX_DIR}/include)
    set(ONNX_AVAILABLE ON)
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNX_DIR}/include)
    # Find the platform-specific library
    if(APPLE)
        find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNX_DIR}/lib NO_DEFAULT_PATH)
    else()
        find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNX_DIR}/lib NO_DEFAULT_PATH)
    endif()
    if(ONNXRUNTIME_LIB)
        message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
    else()
        set(ONNX_AVAILABLE OFF)
        message(STATUS "ONNX Runtime library not found — Emotion model will use stub")
    endif()
else()
    set(ONNX_AVAILABLE OFF)
    message(STATUS "ONNX Runtime not found — Emotion model will use stub")
endif()

# ============================================
# Main engine target
# ============================================
set(ENGINE_SOURCES
    src/main.cpp
    src/pipeline.cpp
    src/vad.cpp
    src/stt.cpp
    src/emotion.cpp
    src/mood_tracker.cpp
    src/llm.cpp
)

add_executable(mindease_engine ${ENGINE_SOURCES})

target_include_directories(mindease_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# ============================================
# Link libraries
# ============================================
set(ENGINE_LIBS "")

if(WHISPER_AVAILABLE)
    target_compile_definitions(mindease_engine PRIVATE HAS_WHISPER=1)
    list(APPEND ENGINE_LIBS whisper)
else()
    target_compile_definitions(mindease_engine PRIVATE HAS_WHISPER=0)
endif()

if(LLAMA_AVAILABLE)
    target_compile_definitions(mindease_engine PRIVATE HAS_LLAMA=1)
    list(APPEND ENGINE_LIBS llama common)
else()
    target_compile_definitions(mindease_engine PRIVATE HAS_LLAMA=0)
endif()

if(ONNX_AVAILABLE)
    target_compile_definitions(mindease_engine PRIVATE HAS_ONNX=1)
    target_include_directories(mindease_engine PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
    list(APPEND ENGINE_LIBS ${ONNXRUNTIME_LIB})
else()
    target_compile_definitions(mindease_engine PRIVATE HAS_ONNX=0)
endif()

target_link_libraries(mindease_engine PRIVATE ${ENGINE_LIBS})

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(mindease_engine PRIVATE Threads::Threads)

# ============================================
# Compiler warnings
# ============================================
target_compile_options(mindease_engine PRIVATE
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:Debug>:-g -O0 -fsanitize=address>
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_options(mindease_engine PRIVATE -fsanitize=address)
endif()

# ============================================
# Install
# ============================================
install(TARGETS mindease_engine DESTINATION bin)

message(STATUS "")
message(STATUS "=== MIND EASE Engine Configuration ===")
message(STATUS "  Metal:   ${MINDEASE_USE_METAL}")
message(STATUS "  CUDA:    ${MINDEASE_USE_CUDA}")
message(STATUS "  Whisper: ${WHISPER_AVAILABLE}")
message(STATUS "  LLaMA:   ${LLAMA_AVAILABLE}")
message(STATUS "  ONNX:    ${ONNX_AVAILABLE}")
message(STATUS "=======================================")
