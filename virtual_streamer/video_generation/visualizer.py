"""
HTML Visualizer for Video Generation Results

This module creates interactive HTML reports showing:
- Each sentence from the story
- Video candidates evaluated for each sentence
- LLM judgement results (rating, grade, reasoning)
- Keywords used for search
- Alternative searches tried
- Final selected video
- Timing information

Usage:
    from virtual_streamer.video_generation import create_html_report

    # From GenerationResult
    html_path = create_html_report(result, output_path="report.html")

    # From ConfigDump
    html_path = create_html_report_from_dump("config_dump.json", output_path="report.html")
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from virtual_streamer.video_generation.config import GenerationResult, ConfigDump


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Generation Report - {title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header {{
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 8px;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .card.green {{
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        }}
        
        .card.blue {{
            background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        }}
        
        .card.orange {{
            background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        }}
        
        .card.purple {{
            background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%);
        }}
        
        .card-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        
        .card-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .sentence-block {{
            background: #f8f9fa;
            border-left: 4px solid #4CAF50;
            padding: 25px;
            margin-bottom: 30px;
            border-radius: 8px;
        }}
        
        .sentence-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .sentence-number {{
            background: #4CAF50;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .sentence-text {{
            font-size: 1.2em;
            color: #2c3e50;
            margin: 15px 0;
            padding: 15px;
            background: white;
            border-radius: 5px;
            font-style: italic;
        }}
        
        .final-selection {{
            background: #e8f5e9;
            border: 2px solid #4CAF50;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        
        .final-selection-header {{
            font-weight: bold;
            color: #2e7d32;
            font-size: 1.1em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }}
        
        .final-selection-header::before {{
            content: "✓";
            background: #4CAF50;
            color: white;
            border-radius: 50%;
            width: 25px;
            height: 25px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-right: 10px;
            font-weight: bold;
        }}
        
        .video-path {{
            font-family: 'Courier New', monospace;
            background: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 0.9em;
            word-break: break-all;
        }}
        
        .candidates-section {{
            margin-top: 20px;
        }}
        
        .section-title {{
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .candidate {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }}
        
        .candidate:hover {{
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        
        .candidate-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .rating-badge {{
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        
        .rating-CONTEXTUAL {{
            background: #4CAF50;
            color: white;
        }}
        
        .rating-NEUTRAL {{
            background: #FF9800;
            color: white;
        }}
        
        .rating-NOT_CONTEXTUAL {{
            background: #f44336;
            color: white;
        }}
        
        .grade {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2196F3;
        }}
        
        .reasoning {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 0.95em;
            color: #555;
            border-left: 3px solid #2196F3;
        }}
        
        .alternatives {{
            background: #fff3e0;
            border: 1px solid #ffb74d;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }}
        
        .alternatives-header {{
            font-weight: bold;
            color: #e65100;
            margin-bottom: 10px;
        }}
        
        .keyword {{
            display: inline-block;
            background: #FF9800;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            margin: 5px;
            font-size: 0.9em;
        }}
        
        .timing-section {{
            background: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin-top: 40px;
        }}
        
        .timing-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .timing-item {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #2196F3;
        }}
        
        .timing-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .timing-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2196F3;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        
        .no-candidates {{
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #f44336;
        }}
        
        .story-section {{
            background: #f3e5f5;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 40px;
            border-left: 4px solid #9C27B0;
        }}
        
        .story-title {{
            font-size: 1.5em;
            font-weight: bold;
            color: #6a1b9a;
            margin-bottom: 10px;
        }}
        
        .story-plan {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            color: #555;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 20px;
            }}
            
            .summary-cards {{
                grid-template-columns: 1fr;
            }}
            
            .sentence-header {{
                flex-direction: column;
                align-items: flex-start;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Video Generation Report</h1>
            <div class="subtitle">Generated on {timestamp}</div>
        </div>
        
        {story_section}
        
        <div class="summary-cards">
            <div class="card green">
                <div class="card-label">Total Sentences</div>
                <div class="card-value">{total_sentences}</div>
            </div>
            <div class="card blue">
                <div class="card-label">Total Duration</div>
                <div class="card-value">{total_duration}s</div>
            </div>
            <div class="card orange">
                <div class="card-label">LLM Calls</div>
                <div class="card-value">{llm_calls}</div>
            </div>
            <div class="card purple">
                <div class="card-label">Total Time</div>
                <div class="card-value">{total_time}s</div>
            </div>
        </div>
        
        {sentences_html}
        
        {timing_section}
        
        <div class="footer">
            <p>Generated by Virtual Streamer Video Generation v1.2.0</p>
            <p>Config Dump: {config_dump_path}</p>
        </div>
    </div>
</body>
</html>
"""


def create_html_report(
    result: GenerationResult, output_path: Optional[str] = None
) -> str:
    """
    Create an HTML visualization report from a GenerationResult.

    Args:
        result: GenerationResult object from video generation
        output_path: Optional path to save HTML file (default: auto-generated)

    Returns:
        Path to the generated HTML file
    """
    # Load config dump if available
    if result.config_dump_path and os.path.exists(result.config_dump_path):
        dump = ConfigDump.load(result.config_dump_path)
        return _create_html_from_dump(dump, result, output_path)
    else:
        # Create basic report from result only
        return _create_basic_html(result, output_path)


def create_html_report_from_dump(
    config_dump_path: str, output_path: Optional[str] = None
) -> str:
    """
    Create an HTML visualization report from a config dump file.

    Args:
        config_dump_path: Path to config dump JSON file
        output_path: Optional path to save HTML file (default: auto-generated)

    Returns:
        Path to the generated HTML file
    """
    dump = ConfigDump.load(config_dump_path)
    return _create_html_from_dump(dump, None, output_path)


def _create_html_from_dump(
    dump: ConfigDump, result: Optional[GenerationResult], output_path: Optional[str]
) -> str:
    """Create HTML from config dump with full details."""

    # Extract data from dump
    sentences = dump.input.get("sentences", [])
    video_matches = dump.execution.get("video_matches", [])
    timing = dump.timing

    # Story section (if available)
    story_section = ""
    story_output = dump.input.get("story_output")
    if story_output:
        story_section = f"""
        <div class="story-section">
            <div class="story-title">📝 {story_output.get("title", "Story")}</div>
            <div class="story-plan">
                <strong>Story Plan:</strong><br>
                {story_output.get("story_plan", "N/A").replace(chr(10), "<br>")}
            </div>
        </div>
        """

    # Generate HTML for each sentence
    sentences_html = []
    llm_call_count = 0

    for idx, (sentence, match_data) in enumerate(zip(sentences, video_matches)):
        sentence_num = idx + 1

        # Extract match data
        selected_video = match_data.get("selected_video", "")
        rating = match_data.get("rating", "UNKNOWN")
        grade = match_data.get("grade", 0)
        reasoning = match_data.get("reasoning", "No reasoning available")
        alternatives = match_data.get("alternatives_tried", [])

        # Count LLM calls (rough estimate)
        llm_call_count += 1  # For video judgement
        if alternatives:
            llm_call_count += len(alternatives)  # For alternative searches

        # Generate alternatives HTML
        alternatives_html = ""
        if alternatives:
            keywords_html = "".join(
                [f'<span class="keyword">{kw}</span>' for kw in alternatives]
            )
            alternatives_html = f"""
            <div class="alternatives">
                <div class="alternatives-header">🔄 Alternative Searches Tried:</div>
                <div>{keywords_html}</div>
            </div>
            """

        # Final selection
        final_selection_html = f"""
        <div class="final-selection">
            <div class="final-selection-header">Selected Video</div>
            <div class="video-path">{selected_video or "No video selected"}</div>
            <div style="margin-top: 10px;">
                <span class="rating-badge rating-{rating}">{rating}</span>
                <span class="grade">Grade: {grade}/10</span>
            </div>
            <div class="reasoning">{reasoning}</div>
        </div>
        """

        # Combine sentence block
        sentence_html = f"""
        <div class="sentence-block">
            <div class="sentence-header">
                <span class="sentence-number">Sentence {sentence_num}</span>
            </div>
            <div class="sentence-text">"{sentence}"</div>
            {final_selection_html}
            {alternatives_html}
        </div>
        """

        sentences_html.append(sentence_html)

    # Timing section
    timing_html = ""
    if timing:
        timing_items = []
        for key, value in timing.items():
            label = key.replace("_", " ").title()
            timing_items.append(f"""
            <div class="timing-item">
                <div class="timing-label">{label}</div>
                <div class="timing-value">{value:.2f}s</div>
            </div>
            """)

        timing_html = f"""
        <div class="timing-section">
            <div class="section-title">⏱️ Performance Metrics</div>
            <div class="timing-grid">
                {"".join(timing_items)}
            </div>
        </div>
        """

    # Get output info
    total_duration = dump.output.get("duration", 0) if hasattr(dump, "output") else 0

    # Generate final HTML
    html_content = HTML_TEMPLATE.format(
        title=story_output.get("title", "Video Generation")
        if story_output
        else "Video Generation",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        story_section=story_section,
        total_sentences=len(sentences),
        total_duration=f"{total_duration:.1f}" if total_duration else "N/A",
        llm_calls=llm_call_count,
        total_time=f"{timing.get('total', 0):.1f}" if timing else "N/A",
        sentences_html="\n".join(sentences_html),
        timing_section=timing_html,
        config_dump_path=os.path.basename(dump.config_dump_path)
        if hasattr(dump, "config_dump_path")
        else "N/A",
    )

    # Determine output path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"video_generation_report_{timestamp}.html"

    # Write HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return os.path.abspath(output_path)


def _create_basic_html(result: GenerationResult, output_path: Optional[str]) -> str:
    """Create basic HTML from result only (without full config dump)."""

    # Basic info from result
    metadata = result.metadata

    story_section = ""
    if result.story_output:
        story_section = f"""
        <div class="story-section">
            <div class="story-title">📝 {result.story_output.title}</div>
            <div class="story-plan">
                <strong>Story Plan:</strong><br>
                {result.story_output.story_plan.replace(chr(10), "<br>")}
            </div>
        </div>
        """

    timing_section = ""
    if "timing" in metadata:
        timing = metadata["timing"]
        timing_items = []
        for key, value in timing.items():
            label = key.replace("_", " ").title()
            timing_items.append(f"""
            <div class="timing-item">
                <div class="timing-label">{label}</div>
                <div class="timing-value">{value:.2f}s</div>
            </div>
            """)

        timing_section = f"""
        <div class="timing-section">
            <div class="section-title">⏱️ Performance Metrics</div>
            <div class="timing-grid">
                {"".join(timing_items)}
            </div>
        </div>
        """

    # Simple sentence list
    sentences_html = f"""
    <div class="sentence-block">
        <div class="section-title">Generated Video</div>
        <div class="final-selection">
            <div class="final-selection-header">Video File</div>
            <div class="video-path">{result.video_path}</div>
        </div>
        <p style="margin-top: 20px; color: #666;">
            <em>For detailed sentence-by-sentence analysis, ensure config_dump is enabled during generation.</em>
        </p>
    </div>
    """

    html_content = HTML_TEMPLATE.format(
        title=result.story_output.title if result.story_output else "Video Generation",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        story_section=story_section,
        total_sentences=metadata.get("sentence_count", "N/A"),
        total_duration=f"{metadata.get('total_duration', 0):.1f}",
        llm_calls="N/A",
        total_time=f"{metadata.get('timing', {}).get('total', 0):.1f}",
        sentences_html=sentences_html,
        timing_section=timing_section,
        config_dump_path=result.config_dump_path or "N/A",
    )

    # Determine output path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"video_generation_report_{timestamp}.html"

    # Write HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return os.path.abspath(output_path)
