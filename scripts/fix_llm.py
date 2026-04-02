#!/usr/bin/env python3
"""Fix the truncated build_prompt function in llm.cpp"""
import pathlib

llm_path = pathlib.Path("/Users/jp710/Desktop/MIND EASE - Mental Wellness by Voice/engine/src/llm.cpp")
content = llm_path.read_text()

# Find where it's truncated
marker = '    // User message\n    oss << "'
idx = content.rfind(marker)
if idx == -1:
    print("Marker not found")
    exit(1)

# Keep everything before the marker, then add the complete ending
prefix = content[:idx]

suffix = '''    // User message
    oss << "
