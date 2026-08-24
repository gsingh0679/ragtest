"""Pytest configuration for tests."""

import sys
from pathlib import Path

# Add parent directory to path so tests can import src
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
