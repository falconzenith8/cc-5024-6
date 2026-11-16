"""浮点数据的读写：.npy、极简 FITS（float32/float64 二维）、PNG 读取。

科学数据一律为浮点；显示用的拉伸参数不属于本模块，也绝不写回数据文件。
"""

from __future__ import annotations

import json
import struct

import numpy as np


def save_npy(path, array):
    np.save(path, np.asarray(array, dtype=np.float32))


def load_npy(path):
    return np.load(path).astype(np.float64)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(obj), f, ensure_ascii=False, indent=2)


def _sanitize(o):
    """NaN/Inf 不是合法 JSON；numpy 标量转 Python 标量。"""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# ---------------- 极简 FITS（仅支持本工具需要的子集） ----------------

def _fits_card(key, value, comment=""):
    if isinstance(value, bool):
        v = "T" if value else "F"
        body = f"{key:<8}= {v:>20}"
    elif isinstance(value, int):
        body = f"{key:<8}= {value:>20d}"
    elif isinstance(value, str):
        body = f"{key:<8}= '{value}'"
    else:
        raise ValueError(key)
    if comment:
        body += f" / {comment}"
    return body[:80].ljust(80)


def write_fits(path_or_file, data):
    """写出二维浮点 FITS（BITPIX=-32/-64，大端）。NaN 原样保留。

    path_or_file 可为路径或可写的二进制文件对象。
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("仅支持二维图像")
    if data.dtype == np.float64:
        bitpix, fmt = -64, ">f8"
        out = data
    else:
        bitpix, fmt = -32, ">f4"
        out = data.astype(np.float32)
    h, w = data.shape
    cards = [
        _fits_card("SIMPLE", True, "conforms to FITS standard"),
        _fits_card("BITPIX", bitpix, "IEEE float"),
        _fits_card("NAXIS", 2),
        _fits_card("NAXIS1", w),
        _fits_card("NAXIS2", h),
        _fits_card("BUNIT", "adu", "flux unit"),
        _fits_card("CTYPE1", "PIXEL", "pixel centers at integer coords"),
        _fits_card("CTYPE2", "PIXEL", "pixel centers at integer coords"),
        "END".ljust(80),
    ]
    header = "".join(cards).encode("ascii")
    header += b" " * ((2880 - len(header) % 2880) % 2880)
    payload = out.astype(fmt).tobytes()
    payload += b"\0" * ((2880 - len(payload) % 2880) % 2880)
    if hasattr(path_or_file, "write"):
        path_or_file.write(header)
        path_or_file.write(payload)
    else:
        with open(path_or_file, "wb") as f:
            f.write(header)
            f.write(payload)


def read_fits(path):
    """读取本工具写出的那类二维浮点 FITS。"""
    with open(path, "rb") as f:
        raw = f.read()
    cards = {}
    pos = 0
    while True:
        card = raw[pos:pos + 80].decode("ascii", "replace")
        pos += 80
        key = card[:8].strip()
        if key == "END":
            break
        if "=" in card[:10]:
            val = card[10:30].strip()
            if val in ("T", "F"):
                cards[key] = val == "T"
            elif val.startswith("'"):
                cards[key] = val.strip("'")
            else:
                cards[key] = int(float(val))
    pos = (pos + 2879) // 2880 * 2880
    bitpix = cards["BITPIX"]
    w, h = cards["NAXIS1"], cards["NAXIS2"]
    if bitpix == -32:
        dtype = ">f4"
    elif bitpix == -64:
        dtype = ">f8"
    else:
        raise ValueError(f"不支持的 BITPIX={bitpix}")
    n = w * h * np.dtype(dtype).itemsize
    data = np.frombuffer(raw[pos:pos + n], dtype=dtype).reshape(h, w)
    return np.asarray(data, dtype=np.float64)


def read_image(path):
    """按扩展名读取 .npy / .fits(.fit) / .png(.jpg 等 Pillow 支持的格式)。

    返回 float64 二维数组。彩色图转灰度（ITU-R 601-2 亮度）。
    """
    p = str(path).lower()
    if p.endswith(".npy"):
        return load_npy(path)
    if p.endswith((".fits", ".fit", ".fts")):
        return read_fits(path)
    from PIL import Image

    img = Image.open(path)
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return arr


def sniff_frames(files):
    """把一组文件读成帧列表；单个 3D .npy 视为帧序列。"""
    frames, names = [], []
    for path in files:
        arr = read_image(path)
        name = getattr(path, "filename", None) or str(path)
        if arr.ndim == 3:
            for i in range(arr.shape[0]):
                frames.append(arr[i])
                names.append(f"{name}#{i}")
        else:
            frames.append(arr)
            names.append(name)
    return frames, names
