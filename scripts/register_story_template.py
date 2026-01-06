#!/usr/bin/env python3
"""
Story Template Registration Script.

Registers a new story template with associated characters via the API.
Supports two cleanup modes:
  --purge: Delete all templates but keep table schema
  --drop-tables: Drop and recreate tables (for schema migrations)

Usage:
    python scripts/register_story_template.py \
        --name "C'est pas Sorcier Parody" \
        --characters fred jamy \
        --collection cps_videos \
        --target-lines 6 \
        --prompt-file prompts/cest_pas_sorcier.txt

    python scripts/register_story_template.py \
        --name "AI Jesus Sermon" \
        --characters jesus \
        --collection jesus_videos \
        --target-lines 8 \
        --prompt "You are creating sermons in the style of AI Jesus..."

    # Drop tables and recreate with new schema, then register
    python scripts/register_story_template.py --drop-tables \
        --name "C'est pas Sorcier" \
        --characters fred jamy \
        --collection cps_videos \
        --prompt-file prompts/cest_pas_sorcier.txt
"""

import argparse
import sys
from pathlib import Path

import requests


def drop_tables(api_url: str) -> bool:
    """
    Drop and recreate the story template tables.
    
    This is useful for schema migrations when the table structure has changed.
    
    Args:
        api_url: Base API URL
        
    Returns:
        True if successful
    """
    url = f"{api_url.rstrip('/')}/api/v1/story-templates/admin/drop-tables"
    
    print("\nDropping and recreating story template tables...")
    print("  WARNING: This deletes ALL story templates permanently!")
    
    response = requests.post(url)
    
    if response.ok:
        result = response.json()
        print(f"✓ {result.get('message', 'Tables reset successfully')}")
        return True
    else:
        print(f"✗ Failed to drop tables: {response.status_code}")
        print(f"  Response: {response.text}")
        return False


def purge_all_templates(api_url: str) -> int:
    """
    Delete all existing story templates (keeps table schema).
    
    Args:
        api_url: Base API URL
        
    Returns:
        Number of templates deleted
    """
    base_url = f"{api_url.rstrip('/')}/api/v1/story-templates"
    
    # First, list all templates
    print("\nFetching existing story templates...")
    response = requests.get(base_url)
    
    if not response.ok:
        print(f"✗ Failed to list templates: {response.status_code}")
        print(f"  Response: {response.text}")
        return 0
    
    templates = response.json()
    
    if not templates:
        print("No existing templates to delete.")
        return 0
    
    print(f"Found {len(templates)} template(s) to delete:")
    for t in templates:
        print(f"  - {t.get('template_id')}: {t.get('name')}")
    
    # Delete each template
    deleted = 0
    for template in templates:
        template_id = template.get("template_id")
        delete_url = f"{base_url}/{template_id}"
        
        del_response = requests.delete(delete_url)
        if del_response.ok or del_response.status_code == 204:
            print(f"  ✓ Deleted: {template_id}")
            deleted += 1
        else:
            print(f"  ✗ Failed to delete {template_id}: {del_response.status_code}")
    
    print(f"\n✓ Purged {deleted}/{len(templates)} template(s)")
    return deleted


def register_via_api(
    api_url: str,
    name: str,
    prompt: str,
    collection: str,
    target_lines: int,
    character_ids: list[str],
) -> dict:
    """
    Register story template via POST /story-templates API.
    
    Args:
        api_url: Base API URL
        name: Template display name
        prompt: Full prompt text for story generation
        collection: Qdrant collection name for video search
        target_lines: Target number of dialogue lines
        character_ids: List of character IDs to associate
        
    Returns:
        API response as dict
    """
    url = f"{api_url.rstrip('/')}/api/v1/story-templates"
    
    # Prepare form data
    data = {
        "name": name,
        "prompt": prompt,
        "collection": collection,
        "target_lines": target_lines,
    }
    
    # FastAPI expects list values as separate form fields
    files = []
    for char_id in character_ids:
        files.append(("character_ids", (None, char_id)))
    
    print(f"\nRegistering story template '{name}' via API...")
    print(f"  URL: {url}")
    print(f"  Collection: {collection}")
    print(f"  Characters: {', '.join(character_ids)}")
    print(f"  Target lines: {target_lines}")
    print(f"  Prompt length: {len(prompt)} chars")
    
    response = requests.post(url, data=data, files=files if files else None)
    
    if response.ok:
        print(f"✓ Story template '{name}' registered successfully!")
        return response.json()
    else:
        print(f"✗ Failed to register story template: {response.status_code}")
        print(f"  Response: {response.text}")
        response.raise_for_status()


def main():
    parser = argparse.ArgumentParser(
        description="Register a story template for video generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register with prompt file
  python scripts/register_story_template.py \\
      --name "C'est pas Sorcier Parody" \\
      --characters fred jamy \\
      --collection cps_videos \\
      --target-lines 6 \\
      --prompt-file prompts/cest_pas_sorcier.txt

  # Register with inline prompt
  python scripts/register_story_template.py \\
      --name "AI Jesus Sermon" \\
      --characters jesus \\
      --collection jesus_videos \\
      --target-lines 8 \\
      --prompt "You are creating sermons..."

  # Purge all existing templates and register a new one
  python scripts/register_story_template.py \\
      --purge \\
      --name "C'est pas Sorcier" \\
      --characters fred jamy \\
      --collection cps_videos \\
      --prompt-file prompts/cest_pas_sorcier.txt

  # Drop tables for schema migration, then register
  python scripts/register_story_template.py \\
      --drop-tables \\
      --name "C'est pas Sorcier" \\
      --characters fred jamy \\
      --collection cps_videos \\
      --prompt-file prompts/cest_pas_sorcier.txt
        """,
    )
    
    # Template identity
    parser.add_argument(
        "--name",
        required=True,
        help="Display name for the story template (also used as template_id)",
    )
    
    # Characters
    parser.add_argument(
        "--characters",
        nargs="+",
        required=True,
        help="Character IDs to associate with this template (must exist in DB)",
    )
    
    # Collection
    parser.add_argument(
        "--collection",
        required=True,
        help="Qdrant collection name for video search",
    )
    
    # Target lines
    parser.add_argument(
        "--target-lines",
        type=int,
        default=6,
        help="Target number of dialogue lines in generated stories (default: 6)",
    )
    
    # Prompt sources (mutually exclusive)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt-file",
        help="Path to file containing the prompt text",
    )
    prompt_group.add_argument(
        "--prompt",
        help="Inline prompt text",
    )
    
    # API settings
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    
    # Cleanup options
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Delete all existing story templates before registering (keeps table schema)",
    )
    parser.add_argument(
        "--drop-tables",
        action="store_true",
        help="Drop and recreate story template tables before registering (for schema migrations)",
    )
    
    args = parser.parse_args()
    
    # Load prompt from file or use inline
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            print(f"Error: Prompt file not found: {args.prompt_file}")
            sys.exit(1)
        
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        print(f"\n{'='*60}")
        print(f"Story Template Registration: {args.name}")
        print(f"{'='*60}")
        print(f"\nLoaded prompt from: {args.prompt_file}")
    else:
        prompt = args.prompt
        print(f"\n{'='*60}")
        print(f"Story Template Registration: {args.name}")
        print(f"{'='*60}")
        print(f"\nUsing inline prompt")
    
    print(f"\nPrompt preview:")
    print("-" * 60)
    # Show first 500 chars of prompt
    preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
    print(preview)
    print("-" * 60)
    
    print(f"\nCollection: {args.collection}")
    print(f"Characters: {', '.join(args.characters)}")
    print(f"Target lines: {args.target_lines}")
    
    # Drop tables if requested (for schema migrations)
    if args.drop_tables:
        print(f"\n{'='*60}")
        print("DROPPING AND RECREATING STORY TEMPLATE TABLES")
        print(f"{'='*60}")
        try:
            success = drop_tables(args.api_url)
            if not success:
                sys.exit(1)
        except requests.exceptions.ConnectionError:
            print(f"\n✗ Error: Could not connect to API at {args.api_url}")
            print("  Make sure the API server is running.")
            sys.exit(1)
        except requests.exceptions.HTTPError as e:
            print(f"\n✗ Error: Drop tables failed: {e}")
            sys.exit(1)
    # Purge existing templates if requested (and not dropping tables)
    elif args.purge:
        print(f"\n{'='*60}")
        print("PURGING ALL EXISTING TEMPLATES")
        print(f"{'='*60}")
        try:
            purge_all_templates(args.api_url)
        except requests.exceptions.ConnectionError:
            print(f"\n✗ Error: Could not connect to API at {args.api_url}")
            print("  Make sure the API server is running.")
            sys.exit(1)
        except requests.exceptions.HTTPError as e:
            print(f"\n✗ Error: Purge failed: {e}")
            sys.exit(1)
    
    # Register via API
    try:
        result = register_via_api(
            api_url=args.api_url,
            name=args.name,
            prompt=prompt,
            collection=args.collection,
            target_lines=args.target_lines,
            character_ids=args.characters,
        )
        
        print(f"\n{'='*60}")
        print("Registration Complete!")
        print(f"{'='*60}")
        print(f"  Template ID: {result.get('template_id')}")
        print(f"  Name: {result.get('name')}")
        print(f"  Collection: {result.get('collection')}")
        print(f"  Target lines: {result.get('target_lines')}")
        print(f"  Characters: {', '.join(result.get('character_ids', []))}")
        
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Error: Could not connect to API at {args.api_url}")
        print("  Make sure the API server is running.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n✗ Error: API request failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

