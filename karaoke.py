#!/usr/bin/env python3
"""Entry point for Y2Karaoke - backward compatibility."""

import sys
from pathlib import Path

# Add src to path so we can import the new modules
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from y2karaoke.cli import cli

if __name__ == "__main__":
    cli()
