#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Main pipeline entry point.
"""
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))

from pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
