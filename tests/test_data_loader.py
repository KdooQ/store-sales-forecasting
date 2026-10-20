import pandas as pd
import pytest

from src.data_loader import EXPECTED_COLUMNS, load_all_tables, load_table

# CSV mínimos (con las mismas columnas que los reales) para no depender de Kaggle
SAMPLE_CSVS = {
    "train": "id,date,store_nbr,family,sales,onpromotion\n0,2013-01-01,1,AUTOMOTIVE,0.0,0\n1,2013-01-01,1,BEVERAGES,12.0,0\n",
    "test": "id,date,store_nbr,family,onpromotion\n3000888,2017-08-16,1,AUTOMOTIVE,0\n",
    "stores": "store_nbr,city,state,type,cluster\n1,Quito,Pichincha,D,13\n",
    "oil": "date,dcoilwtico\n2013-01-01,\n2013-01-02,93.14\n",
    "holidays_events": "date,type,locale,locale_name,description,transferred\n2012-12-25,Holiday,National,Ecuador,Navidad,False\n",
    "transactions": "date,store_nbr,transactions\n2013-01-01,25,770\n",
}


@pytest.fixture
def raw_dir(tmp_path):
    """Carpeta temporal con un CSV mínimo por cada tabla."""
    for name, content in SAMPLE_CSVS.items():
        (tmp_path / f"{name}.csv").write_text(content, encoding="utf-8")
    return tmp_path


def test_load_table_parses_dates(raw_dir):
    df = load_table("oil", raw_dir)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_table_returns_expected_columns(raw_dir):
    df = load_table("train", raw_dir)
    assert list(df.columns) == EXPECTED_COLUMNS["train"]


def test_load_table_raises_when_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_table("oil", tmp_path)


def test_load_table_raises_when_columns_are_missing(raw_dir):
    (raw_dir / "oil.csv").write_text("date,precio\n2013-01-01,93.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dcoilwtico"):
        load_table("oil", raw_dir)


def test_load_table_raises_when_table_is_empty(raw_dir):
    (raw_dir / "oil.csv").write_text("date,dcoilwtico\n", encoding="utf-8")
    with pytest.raises(ValueError, match="vacía"):
        load_table("oil", raw_dir)


def test_load_table_rejects_unknown_table(raw_dir):
    with pytest.raises(ValueError, match="desconocida"):
        load_table("ventas", raw_dir)


def test_load_all_tables_returns_every_table(raw_dir):
    tables = load_all_tables(raw_dir)
    assert set(tables) == set(EXPECTED_COLUMNS)