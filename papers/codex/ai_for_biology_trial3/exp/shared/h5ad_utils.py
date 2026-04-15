from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class H5adData:
    x: sparse.csr_matrix
    obs: pd.DataFrame
    var: pd.DataFrame
    uns: dict[str, Any]


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O"}:
        return np.array([_decode(v) for v in value])
    return value


def _read_dataset(node: h5py.Dataset) -> Any:
    value = node[()]
    if isinstance(value, np.ndarray):
        return _decode(value)
    return _decode(value)


def _read_categorical(group: h5py.Group) -> pd.Categorical:
    codes = group["codes"][()]
    cats = _decode(group["categories"][()])
    return pd.Categorical.from_codes(codes.astype(int), cats)


def _read_frame(group: h5py.Group) -> pd.DataFrame:
    columns = []
    data = {}
    if "_index" in group:
        index_name = _decode(group["_index"][()])
        index_values = _decode(group[index_name][()])
    else:
        index_name = None
        index_values = np.arange(group.attrs["shape"][0])
    for key in group.keys():
        if key == "_index" or key == index_name:
            continue
        node = group[key]
        if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
            value = _read_categorical(node)
        elif isinstance(node, h5py.Dataset):
            value = _read_dataset(node)
        else:
            continue
        data[key] = value
        columns.append(key)
    frame = pd.DataFrame(data, columns=columns)
    frame.index = pd.Index(index_values, name=index_name or "index")
    return frame


def _read_sparse(group: h5py.Group) -> sparse.csr_matrix:
    data = group["data"][()]
    indices = group["indices"][()]
    indptr = group["indptr"][()]
    shape = tuple(group["shape"][()].tolist())
    encoding = group.attrs.get("encoding-type", b"csr_matrix")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    if encoding == "csr_matrix":
        return sparse.csr_matrix((data, indices, indptr), shape=shape)
    if encoding == "csc_matrix":
        return sparse.csc_matrix((data, indices, indptr), shape=shape).tocsr()
    raise ValueError(f"Unsupported sparse encoding: {encoding}")


def _read_uns(node: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(node, h5py.Dataset):
        return _read_dataset(node)
    out = {}
    for key, child in node.items():
        out[key] = _read_uns(child)
    return out


def read_h5ad(path: str | Path) -> H5adData:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        x_node = handle["X"]
        if isinstance(x_node, h5py.Group):
            x = _read_sparse(x_node)
        else:
            x = sparse.csr_matrix(x_node[()])
        obs = _read_frame(handle["obs"])
        var = _read_frame(handle["var"])
        uns = _read_uns(handle["uns"]) if "uns" in handle else {}
    return H5adData(x=x, obs=obs, var=var, uns=uns)
