import polars as pl
import os
import sys

def load_tsv(filepath: str) -> pl.DataFrame:
    """Fast TSV reader using Polars with explicit tab separator and no quote escaping issues."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pl.read_csv(filepath, separator="\t", quote_char=None)

def load_dataset(data_dir: str, is_train: bool = True):
    """Load S1, S2, S3, and ground truth (if train) dataframes."""
    subfolder = "train" if is_train else "test"
    prefix = "train" if is_train else "test"
    
    path_s1 = os.path.join(data_dir, subfolder, f"{prefix}_source1.tsv")
    path_s2 = os.path.join(data_dir, subfolder, f"{prefix}_source2.tsv")
    path_s3 = os.path.join(data_dir, subfolder, f"{prefix}_source3.tsv")
    
    df_s1 = load_tsv(path_s1)
    df_s2 = load_tsv(path_s2)
    df_s3 = load_tsv(path_s3)
    
    df_gt = None
    if is_train:
        path_gt = os.path.join(data_dir, subfolder, "train_ground_truth.tsv")
        df_gt = load_tsv(path_gt)
        
    return df_s1, df_s2, df_s3, df_gt
