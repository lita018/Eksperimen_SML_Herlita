"""
Automate Preprocessing - Heart Disease Dataset
Nama: Herlita
Username: lita018

File ini melakukan preprocessing otomatis dari data mentah
hingga menghasilkan data yang siap dilatih.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset dari file CSV."""
    logger.info(f"Loading data dari: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data berhasil dimuat. Shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values dengan mengisi menggunakan median."""
    logger.info("Handling missing values...")
    df_clean = df.copy()

    missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
    for col in missing_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        logger.info(f"  - Kolom '{col}': diisi dengan median = {median_val:.2f}")

    total_missing = df_clean.isnull().sum().sum()
    logger.info(f"Missing values setelah handling: {total_missing}")
    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus baris duplikat."""
    logger.info("Menghapus duplikat...")
    before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df_clean)
    logger.info(f"  - Duplikat dihapus: {removed} baris")
    return df_clean


def handle_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Hapus outlier menggunakan metode IQR."""
    logger.info("Handling outliers dengan IQR...")
    df_clean = df.copy()

    for col in cols:
        if col not in df_clean.columns:
            continue
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(df_clean)
        df_clean = df_clean[
            (df_clean[col] >= lower) & (df_clean[col] <= upper)
        ].reset_index(drop=True)
        removed = before - len(df_clean)
        logger.info(f"  - '{col}': {removed} outlier dihapus (range: {lower:.1f} - {upper:.1f})")

    return df_clean


def encode_features(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """One-hot encoding untuk fitur kategorikal."""
    logger.info(f"Encoding fitur kategorikal: {cat_cols}")
    df_clean = df.copy()

    # Convert thal to int if exists
    if 'thal' in df_clean.columns:
        df_clean['thal'] = df_clean['thal'].fillna(df_clean['thal'].median()).astype(int)

    df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
    logger.info(f"Shape setelah encoding: {df_encoded.shape}")
    return df_encoded


def scale_features(df: pd.DataFrame, target_col: str) -> tuple:
    """Scaling fitur numerik menggunakan StandardScaler."""
    logger.info("Scaling fitur...")
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    logger.info("Scaling selesai.")
    return X_scaled, y, scaler


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Train-test split dengan stratifikasi."""
    logger.info(f"Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"  - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test,
                            output_dir: str = 'heart_disease_preprocessing'):
    """Simpan data hasil preprocessing."""
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df['target'] = y_train.values
    test_df = X_test.copy()
    test_df['target'] = y_test.values

    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Data tersimpan di '{output_dir}':")
    logger.info(f"  - train.csv: {train_df.shape}")
    logger.info(f"  - test.csv: {test_df.shape}")

    return train_path, test_path


def preprocess(input_filepath: str,
               output_dir: str = 'heart_disease_preprocessing',
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Fungsi utama preprocessing end-to-end.

    Args:
        input_filepath: Path ke file CSV raw data
        output_dir: Direktori output data preprocessing
        test_size: Proporsi data test (default 0.2)
        random_state: Random seed untuk reproducibility

    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    logger.info("=" * 50)
    logger.info("MULAI PREPROCESSING OTOMATIS - Heart Disease")
    logger.info("=" * 50)

    # 1. Load data
    df = load_data(input_filepath)

    # 2. Handle missing values
    df = handle_missing_values(df)

    # 3. Remove duplicates
    df = remove_duplicates(df)

    # 4. Handle outliers
    outlier_cols = ['chol', 'trestbps', 'oldpeak']
    df = handle_outliers(df, outlier_cols)

    # 5. Encode categorical features
    cat_cols = ['cp', 'restecg', 'slope', 'thal']
    df = encode_features(df, cat_cols)

    # 6. Scale features
    X, y, scaler = scale_features(df, target_col='target')

    # 7. Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    # 8. Save results
    save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir)

    logger.info("=" * 50)
    logger.info("PREPROCESSING SELESAI!")
    logger.info("=" * 50)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automate Preprocessing Heart Disease')
    parser.add_argument('--input', type=str,
                        default='../heart_disease_raw.csv',
                        help='Path ke file CSV raw data')
    parser.add_argument('--output', type=str,
                        default='heart_disease_preprocessing',
                        help='Direktori output')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proporsi data test')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    X_train, X_test, y_train, y_test = preprocess(
        input_filepath=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    )
