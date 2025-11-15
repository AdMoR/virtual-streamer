#!/usr/bin/env python3
"""
Create HTML Visualization Report from Config Dump

This script generates an interactive HTML report showing the complete
video generation process with detailed information about:
- Each sentence and its video candidates
- LLM judgement results (rating, grade, reasoning)
- Alternative keywords tried
- Final video selections
- Performance timing

USAGE:

# Generate report from config dump
python scripts/create_report.py output/config_20251111_173000.json

# Specify output location
python scripts/create_report.py output/config_20251111_173000.json --output report.html

# Auto-find latest config dump
python scripts/create_report.py --latest

# Open report in browser after generation
python scripts/create_report.py output/config.json --open
"""

import sys
import os
import argparse
import webbrowser
from pathlib import Path
from glob import glob

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from virtual_streamer.video_generation import create_html_report_from_dump


def find_latest_config_dump(output_dir: str = "./output") -> str:
    """Find the most recent config dump file."""
    pattern = os.path.join(output_dir, "config_*.json")
    files = glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No config dump files found in {output_dir}")
    
    # Sort by modification time, most recent first
    latest = max(files, key=os.path.getmtime)
    return latest


def main():
    parser = argparse.ArgumentParser(
        description="Create HTML visualization report from video generation config dump",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "config_dump",
        nargs="?",
        help="Path to config dump JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output HTML file path (default: auto-generated)"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest config dump from ./output directory"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the report in web browser after generation"
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory to search for config dumps (default: ./output)"
    )
    
    args = parser.parse_args()
    
    # Determine config dump file
    if args.latest:
        try:
            config_dump_path = find_latest_config_dump(args.output_dir)
            print(f"Using latest config dump: {config_dump_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    elif args.config_dump:
        config_dump_path = args.config_dump
    else:
        print("Error: Please provide a config dump file or use --latest", file=sys.stderr)
        parser.print_help()
        return 1
    
    # Verify file exists
    if not os.path.exists(config_dump_path):
        print(f"Error: Config dump file not found: {config_dump_path}", file=sys.stderr)
        return 1
    
    try:
        # Generate HTML report
        print(f"\n🎬 Generating HTML report from: {config_dump_path}")
        
        html_path = create_html_report_from_dump(
            config_dump_path,
            output_path=args.output
        )
        
        print(f"✓ HTML report created: {html_path}")
        
        # Get file size
        file_size = os.path.getsize(html_path) / 1024  # KB
        print(f"  File size: {file_size:.1f} KB")
        
        # Open in browser if requested
        if args.open:
            print(f"\n🌐 Opening report in web browser...")
            webbrowser.open(f"file://{html_path}")
        else:
            print(f"\n💡 To view the report, open: {html_path}")
            print(f"   Or run: python scripts/create_report.py {config_dump_path} --open")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error generating report: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())




