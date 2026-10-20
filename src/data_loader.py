"""Carga y validación de los CSV crudos de Store Sales."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Tabla (nombre del CSV sin extensión) -> columnas que esperamos encontrar
EXPECTED_COLUMNS = {
    "train": ["id", "date", "store_nbr", "family", "sales", "onpromotion"],
    "test": ["id", "date", "store_nbr", "family", "onpromotion"],
    "stores": ["store_nbr", "city", "state", "type", "cluster"],
    "oil": ["date", "dcoilwtico"],
    "holidays_events": ["date", "type", "locale", "locale_name", "description", "transferred"],
    "transactions": ["date", "store_nbr", "transactions"],
}


def _validate(df: pd.DataFrame, name: str) -> None:
    """Falla con un mensaje claro si faltan columnas o la tabla está vacía."""
    missing = [col for col in EXPECTED_COLUMNS[name] if col not in df.columns]
    if missing:
        raise ValueError(f"La tabla '{name}' no tiene las columnas: {missing}")
    if df.empty:
        raise ValueError(f"La tabla '{name}' está vacía")


def load_table(name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Lee un CSV crudo, valida sus columnas y convierte 'date' a datetime."""
    if name not in EXPECTED_COLUMNS:
        raise ValueError(f"Tabla desconocida: '{name}'. Opciones: {list(EXPECTED_COLUMNS)}")

    path = Path(raw_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No encontré {path}. ¿Pusiste los CSV de Kaggle en {raw_dir}?")

    df = pd.read_csv(path)
    _validate(df, name)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_all_tables(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Carga todas las tablas y las devuelve como {nombre: DataFrame}."""
    return {name: load_table(name, raw_dir) for name in EXPECTED_COLUMNS}