import os
import zipfile
import sys
import time

def make_submission_zip(zip_name="amazon_ml_submission.zip"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(base_dir, zip_name)
    
    # Files and folders to include
    include_map = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("Documentation_template.md", "Documentation_template.md"),
    ]
    
    code_dir = os.path.join(base_dir, "code", "business_entity_resolution")
    for root, dirs, files in os.walk(code_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
        for file in files:
            if file.endswith((".pyc", ".pyo", ".DS_Store")):
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, base_dir)
            arc_path = rel_path.replace("\\", "/")
            include_map.append((rel_path, arc_path))
            
    print(f"Creating submission zip: {zip_path}")
    start_t = time.time()
    total_size = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src_rel, arc_name in include_map:
            src_full = os.path.join(base_dir, src_rel)
            if not os.path.exists(src_full):
                print(f"ERROR: Missing required file {src_full}")
                sys.exit(1)
            file_size = os.path.getsize(src_full)
            total_size += file_size
            print(f"  Adding {arc_name} ({file_size / (1024*1024):.2f} MB)...")
            zf.write(src_full, arcname=arc_name)
            
    zip_size = os.path.getsize(zip_path)
    print(f"\nZip created successfully in {time.time() - start_t:.1f}s!")
    print(f"Uncompressed size: {total_size / (1024*1024):.2f} MB")
    print(f"Compressed zip size: {zip_size / (1024*1024):.2f} MB")
    
    print("\nVerifying archive contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        for name in namelist:
            info = zf.getinfo(name)
            print(f"  - {name} ({info.file_size / (1024*1024):.2f} MB uncompressed)")
        print(f"Total files in zip: {len(namelist)}")

if __name__ == "__main__":
    make_submission_zip()
