# coding: utf-8
"""
Collection of helper functions for audio processing
"""
import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from joeynmt.helpers_for_ddp import get_logger

logger = get_logger(__name__)


# from fairseq
def _is_npy_data(data: bytes) -> bool:
    return data[0] == 147 and data[1] == 78


# from fairseq
def _get_features_from_zip(path, byte_offset, byte_size):
    with path.open("rb") as f:
        f.seek(byte_offset)
        data = f.read(byte_size)
    byte_features = io.BytesIO(data)
    if len(data) > 1 and _is_npy_data(data):
        features = np.load(byte_features)
    else:
        raise ValueError(
            f'Unknown file format for '
            f'"{path}" [{byte_offset}:{byte_size}]'
        )
    return features


# from fairseq
def get_n_frames(wave_length: int, sample_rate: int):
    duration_ms = int(wave_length / sample_rate * 1000)
    n_frames = int(1 + (duration_ms - 25) / 10)
    return n_frames


# from fairseq
def get_features(root_path: Path, fbank_path: str) -> np.ndarray:
    """Get speech features from ZIP file
       accessed via byte offset and length

    :return: (np.ndarray) speech features in shape of (num_frames, num_freq)
    """
    _path, *extra = fbank_path.split(":")
    _path = root_path / _path
    if not _path.is_file():
        raise FileNotFoundError(f"File not found: {_path}")

    if len(extra) == 0:
        if _path.suffix == ".npy":
            features = np.load(_path.as_posix())
        elif _path.suffix in [".mp3", ".wav"]:
            waveform, sample_rate = torchaudio.load(_path.as_posix())
            features = extract_fbank_features(waveform, sample_rate)
        else:
            raise ValueError(f"Invalid file type: {_path}")
    elif len(extra) == 2:
        assert _path.suffix == ".zip"
        extra = [int(i) for i in extra]
        features = _get_features_from_zip(_path, extra[0], extra[1])
    else:
        raise ValueError(f"Invalid path: {root_path / fbank_path}")

    assert len(features.shape) == 2, "spectrogram must be a 2-D array."
    return features


def pad_features(
    feat_list: List[np.ndarray],
    embed_size: int = 80,
    pad_index: int = 1,
) -> Tuple[np.ndarray, List[int], None]:
    """
    Pad continuous feature representation in batch.
    called in batch construction (not in data loading)

    :param feat_list: list of features
    :param embed_size: (int) number of frequencies
    :param pad_index: pad index
    :returns:
      - features np.ndarray, (batch_size, src_len, embed_size)
      - lengths List[int], (batch_size)
    """
    max_len = max([int(f.shape[0]) for f in feat_list])
    batch_size = len(feat_list)

    # encoder input has shape of (batch_size, src_len, embed_size)
    # (see encoder.forward())
    features = np.zeros((batch_size, max_len, embed_size), dtype=np.float32)
    features.fill(float(pad_index))
    lengths = []

    for i, f in enumerate(feat_list):
        length = min(int(f.shape[0]), max_len)
        assert length > 0, "empty feature!"
        features[i, :length, :] = f[:length, :]
        lengths.append(length)

    m = max(lengths)
    if m < features.shape[1]:
        features = features[:, :m, :]

    # validation
    assert max(lengths) == features.shape[1]
    assert embed_size == features.shape[2]
    assert sum(lengths) > 0

    return features, lengths, None