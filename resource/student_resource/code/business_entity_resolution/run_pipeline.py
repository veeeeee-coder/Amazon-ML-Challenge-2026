import sys
import os

# Add src to python path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pipeline import run_end_to_end_pipeline

if __name__ == "__main__":
    # Base directory containing dataset/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "dataset")
    output_dir = os.path.join(base_dir, "output")
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    run_end_to_end_pipeline(data_dir=data_dir, output_dir=output_dir)
