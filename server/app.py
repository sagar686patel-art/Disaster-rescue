"""
Server module for multi-mode deployment compatibility.
Exposes the main FastAPI app for HuggingFace Spaces.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the main app from root app.py
from app import app, main

# Export for deployment
__all__ = ["app", "main"]