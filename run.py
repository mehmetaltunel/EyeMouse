#!/usr/bin/env python3
"""
Eye Mouse - Entry Point
Gaze-controlled cursor application
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gui.main_window import main

if __name__ == "__main__":
    main()
