#!/usr/bin/env python3
"""
Validation script to check if all imports work correctly.
Run this before running the actual video generation script.
"""

import sys
import importlib


def check_import(module_name, package=None):
    """Try to import a module and report status."""
    try:
        if package:
            importlib.import_module(module_name, package)
        else:
            importlib.import_module(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False


def main():
    """Check all required imports."""
    print("Checking imports for video generation script...")
    print("=" * 70)

    all_ok = True

    # Core Python modules
    print("\nCore Python modules:")
    for module in ["asyncio", "argparse", "json", "os", "sys", "tempfile"]:
        all_ok &= check_import(module)

    # Pydantic and settings
    print("\nConfiguration (pydantic):")
    for module in ["pydantic", "pydantic_settings", "yaml"]:
        all_ok &= check_import(module)

    # LLM providers
    print("\nLLM providers:")
    for module in ["anthropic", "openai", "litellm"]:
        all_ok &= check_import(module)

    # Video/Audio processing
    print("\nVideo/Audio processing:")
    for module in ["cv2", "stable_whisper"]:
        all_ok &= check_import(module)

    # LlamaIndex
    print("\nLlamaIndex:")
    for module in [
        "llama_index.core",
        "llama_index.retrievers.bm25",
        "llama_index.embeddings.huggingface",
    ]:
        all_ok &= check_import(module)

    # Stemmer
    print("\nOther dependencies:")
    for module in ["Stemmer"]:
        all_ok &= check_import(module)

    # Testing
    print("\nTesting (optional):")
    for module in ["pytest", "pytest_asyncio"]:
        check_import(module)  # Don't fail on test deps

    # Check custom modules
    print("\nCustom modules:")
    sys.path.insert(0, "/Users/amorvan/Documents/code_dw/virtual-streamer")
    for module in [
        "scripts.video_generation_config",
        "scripts.video_generation_interfaces",
        "scripts.video_generation_impl",
        "scripts.video_generation_core",
    ]:
        all_ok &= check_import(module)

    print("\n" + "=" * 70)
    if all_ok:
        print("✓ All required imports successful!")
        print("\nYou can now run:")
        print("  python scripts/generate_video.py --help")
        return 0
    else:
        print("✗ Some imports failed. Install missing dependencies:")
        print("  pip install -r requirements_video_generation.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
