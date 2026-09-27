import polars as pl
import os

DATA_DIR = "dataset"
print("Loading data with Polars...")
train_s1 = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), separator="\t", quote_char=None).select(["entity_id", "country"])
train_s2 = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source2.tsv"), separator="\t", quote_char=None).select(["entity_id", "country"])
train_s3 = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source3.tsv"), separator="\t", quote_char=None).select(["entity_id", "country"])
train_gt = pl.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), separator="\t", quote_char=None)

# Combine S2 and S3 country lookup
s23_country = pl.concat([train_s2, train_s3])

# Explode ground truth matched_entity_ids
gt_exploded = (
    train_gt
    .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
    .with_columns(pl.col("matched_entity_ids").str.split(","))
    .explode("matched_entity_ids")
    .with_columns(pl.col("matched_entity_ids").str.strip_chars())
    .filter(pl.col("matched_entity_ids") != "")
)

# Join S1 country
gt_with_countries = (
    gt_exploded
    .join(train_s1, left_on="source1_entity_id", right_on="entity_id", how="inner")
    .rename({"country": "s1_country"})
    .join(s23_country, left_on="matched_entity_ids", right_on="entity_id", how="inner")
    .rename({"country": "matched_country"})
)

total_pairs = len(gt_with_countries)
cross_country = gt_with_countries.filter(pl.col("s1_country") != pl.col("matched_country"))
diff_count = len(cross_country)

print(f"Total matching pairs: {total_pairs}", flush=True)
print(f"Pairs with DIFFERENT country: {diff_count}", flush=True)
