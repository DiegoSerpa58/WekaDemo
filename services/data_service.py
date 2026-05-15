from __future__ import annotations

from io import BytesIO
from typing import Any

import arff
import numpy as np
import pandas as pd
from fastapi import UploadFile


SUPPORTED_EXTENSIONS = {".csv", ".arff"}


def _decode_bytes(content: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _parse_arff(content: bytes) -> pd.DataFrame:
    parsed: dict[str, Any] = arff.loads(_decode_bytes(content))
    attributes = [name for name, _ in parsed.get("attributes", [])]
    rows = parsed.get("data", [])
    if not attributes or not rows:
        raise ValueError("ARFF file is empty or invalid.")

    dataframe = pd.DataFrame(rows, columns=attributes)
    dataframe.replace("?", np.nan, inplace=True)
    return dataframe


def load_dataset_from_upload(upload: UploadFile, content: bytes) -> pd.DataFrame:
    filename = (upload.filename or "dataset.csv").lower()

    if filename.endswith(".csv"):
        dataset = pd.read_csv(BytesIO(content))
    elif filename.endswith(".arff"):
        dataset = _parse_arff(content)
    else:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file extension. Use: {supported}.")

    if dataset.empty:
        raise ValueError("Dataset is empty.")

    dataset.columns = [str(column).strip() for column in dataset.columns]
    return dataset
