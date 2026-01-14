#!/usr/bin/env python3
"""Migration script to move from old structure to new structure."""

import shutil
import sys
from pathlib import Path

def migrate_project():
    """Migrate the project to new structure."""
    
    print("🔄 Migrating Y2Karaoke to new structure...")
    
    # Check if we're in the right directory
    if not Path("karaoke.py").exists():
        print("❌ Please run this script from the y2karaoke project root")
        sys.exit(1)
    
    # Create backup of old files
    backup_dir = Path("backup_old_structure")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    
    backup_dir.mkdir()
    
    # Files to backup
    old_files = [
        "karaoke.py", "downloader.py", "separator.py", "lyrics.py",
        "renderer.py", "uploader.py", "audio_effects.py", "backgrounds.py"
    ]
    
    print("📦 Creating backup of old files...")
    for file in old_files:
        if Path(file).exists():
            shutil.copy2(file, backup_dir / file)
    
    # Create new entry point that uses the new structure
    entry_point = """#!/usr/bin/env python3
\"\"\"Entry point for Y2Karaoke - backward compatibility.\"\"\"

import sys
from pathlib import Path

# Add src to path so we can import the new modules
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from y2karaoke.cli import cli

if __name__ == "__main__":
    cli()
"""
    
    with open("karaoke.py", "w") as f:
        f.write(entry_point)
    
    print("✅ Migration completed!")
    print(f"📁 Old files backed up to: {backup_dir}")
    print("\n🚀 You can now use the new CLI:")
    print("   python -m y2karaoke.cli generate <youtube_url>")
    print("   OR")
    print("   python karaoke.py generate <youtube_url>")
    print("\n📖 Run 'python karaoke.py --help' to see all new options")

if __name__ == "__main__":
    migrate_project()
