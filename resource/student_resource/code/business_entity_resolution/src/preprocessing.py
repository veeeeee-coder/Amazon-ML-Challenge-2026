import polars as pl

def preprocess_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    High-throughput vectorized preprocessing preserving all raw fields.
    Adds:
    - business_name_normalized
    - business_address_normalized
    - country_normalized
    """
    clean_df = df.with_columns([
        pl.col("business_name").fill_null("").alias("business_name_raw"),
        pl.col("business_address").fill_null("").alias("business_address_raw"),
        pl.col("country").fill_null("UNKNOWN").alias("country_raw"),
        
        pl.col("business_name").fill_null("").str.to_lowercase()
          .str.replace_all("&", " and ")
          .str.replace_all(r'[^\w\s]', " ")
          .str.replace_all(r'\s+', " ")
          .str.strip_chars()
          .alias("business_name_normalized"),
          
        pl.col("business_address").fill_null("").str.to_lowercase()
          .str.replace_all(r'[^\w\s]', " ")
          .str.replace_all(r'\s+', " ")
          .str.strip_chars()
          .alias("business_address_normalized"),
          
        pl.col("country").fill_null("UNKNOWN").str.to_uppercase().str.strip_chars().alias("country_normalized")
    ])
    return clean_df
 