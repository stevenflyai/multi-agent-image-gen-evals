"""Pytest configuration — add project root to sys.path so tests can import source modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
