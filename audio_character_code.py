#!/usr/bin/env python3
"""Audio Spectrum Character Code: an offline experimental prototype.

This prototype stores confirmed audio examples and performs nearest-neighbour
recognition for three independent labels:

    --content  : what was spoken
    --speaker  : who spoke
    --emotion  : the expressed vocal style

It is deliberately conservative: adding a confirmed example extends the
experience database; it does not silently rewrite a general-purpose model.
Only local WAV files are read and SQLite is used for local storage.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


TARGET_SAMPLE_RATE = 16_000
N_FFT = 512
HOP_LENGTH = 160
TARGET_FRAMES = 32
TARGET_FREQ_BINS = 128
CODE_VERSION = "0.1"


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int


def read_wav(path: Path) -> AudioData:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("8-bit、16-bit、または32-bit PCM WAVだけを扱えます")

    if channels < 1:
        raise ValueError("WAVのチャンネル数が不正です")

    data = data.reshape(-1, channels).mean(axis=1)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        data = data / peak

    return AudioData(data.astype(np.float32), sample_rate)


def resample_linear(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    if source_rate == target_rate or samples.size < 2:
        return samples.astype(np.float32, copy=False)

    new_size = max(1, round(samples.size * target_rate / source_rate))

    old_x = np.linspace(
        0.0,
        1.0,
        samples.size,
        endpoint=False,
    )

    new_x = np.linspace(
        0.0,
        1.0,
        new_size,
        endpoint=False,
    )

    return np.interp(
        new_x,
        old_x,
        samples,
    ).astype(np.float32)


def frame_signal(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        samples = np.zeros(N_FFT, dtype=np.float32)

    if samples.size < N_FFT:
        samples = np.pad(
            samples,
            (0, N_FFT - samples.size),
        )

    frame_count = 1 + max(
        0,
        (samples.size - N_FFT) // HOP_LENGTH,
    )

    needed = N_FFT + (frame_count - 1) * HOP_LENGTH

    if samples.size < needed:
        samples = np.pad(
            samples,
            (0, needed - samples.size),
        )

    starts = np.arange(frame_count) * HOP_LENGTH

    frames = np.stack(
        [
            samples[start:start + N_FFT]
            for start in starts
        ]
    )

    window = np.hanning(N_FFT).astype(np.float32)

    return frames * window


def resize_time(
    matrix: np.ndarray,
    target: int,
) -> np.ndarray:
    if matrix.shape[0] == target:
        return matrix.astype(np.float32, copy=False)

    old_x = np.linspace(
        0.0,
        1.0,
        matrix.shape[0],
    )

    new_x = np.linspace(
        0.0,
        1.0,
        target,
    )

    return np.stack(
        [
            np.interp(
                new_x,
                old_x,
                matrix[:, i],
            )
            for i in range(matrix.shape[1])
        ],
        axis=1,
    )


def resize_frequency(
    matrix: np.ndarray,
    target: int,
) -> np.ndarray:
    if matrix.shape[1] == target:
        return matrix.astype(np.float32, copy=False)

    old_x = np.linspace(
        0.0,
        1.0,
        matrix.shape[1],
    )

    new_x = np.linspace(
        0.0,
        1.0,
        target,
    )

    return np.stack(
        [
            np.interp(
                new_x,
                old_x,
                row,
            )
            for row in matrix
        ],
        axis=0,
    )


def extract_code(path: Path) -> dict:
    original = read_wav(path)

    samples = resample_linear(
        original.samples,
        original.sample_rate,
        TARGET_SAMPLE_RATE,
    )

    frames = frame_signal(samples)

    complex_stft = np.fft.rfft(
        frames,
        axis=1,
    )

    magnitude = np.abs(
        complex_stft
    ).astype(np.float32)

    phase = np.angle(
        complex_stft
    ).astype(np.float32)

    # Log magnitude keeps quieter structures visible
    # and reduces raw-volume bias.
    log_magnitude = np.log1p(magnitude)

    max_value = (
        float(np.max(log_magnitude))
        if log_magnitude.size
        else 1.0
    )

    if max_value > 0:
        log_magnitude = log_magnitude / max_value

    resized = resize_frequency(
        resize_time(
            log_magnitude,
            TARGET_FRAMES,
        ),
        TARGET_FREQ_BINS,
    )

    rms = np.sqrt(
        np.mean(frames * frames, axis=1)
        + 1e-12
    )

    rms = resize_time(
        rms[:, None],
        TARGET_FRAMES,
    )[:, 0]

    rms_max = (
        float(np.max(rms))
        if rms.size
        else 1.0
    )

    if rms_max > 0:
        rms = rms / rms_max

    # The first prototype classifies with
    # magnitude plus temporal energy.
    vector = np.concatenate(
        [
            resized.reshape(-1),
            rms,
        ]
    ).astype(np.float32)

    norm = float(np.linalg.norm(vector))

    if norm > 0:
        vector /= norm

    return {
        "version": CODE_VERSION,
        "sample_rate": TARGET_SAMPLE_RATE,
        "original_sample_rate": original.sample_rate,
        "duration_seconds": round(
            float(samples.size / TARGET_SAMPLE_RATE),
            6,
        ),
        "frame_count": int(frames.shape[0]),
        "feature_shape": [int(vector.size)],
        "spectrum_shape": [
            int(resized.shape[0]),
            int(resized.shape[1]),
        ],
        "vector": vector,
        "spectrum": resized.astype(np.float32),
        "phase": phase,
        "energy": rms.astype(np.float32),
    }


def array_blob(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()

    np.save(
        buffer,
        array,
        allow_pickle=False,
    )

    return buffer.getvalue()


def blob_array(blob: bytes) -> np.ndarray:
    return np.load(
        io.BytesIO(blob),
        allow_pickle=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source_path TEXT NOT NULL,
            sha256 TEXT NOT NULL UNIQUE,
            speaker TEXT,
            content TEXT,
            emotion TEXT,
            sample_rate INTEGER NOT NULL,
            duration_seconds REAL NOT NULL,
            feature_shape TEXT NOT NULL,
            vector BLOB NOT NULL,
            spectrum BLOB NOT NULL,
            phase BLOB NOT NULL,
            energy BLOB NOT NULL,
            confirmed INTEGER NOT NULL DEFAULT 1
        )
        """
    )

    connection.commit()

    return connection


def add_sample(
    db_path: Path,
    audio_path: Path,
    speaker: str | None,
    content: str | None,
    emotion: str | None,
) -> None:
    code = extract_code(audio_path)
    digest = sha256_file(audio_path)
    connection = connect(db_path)

    try:
        connection.execute(
            """
            INSERT INTO samples
            (
                created_at,
                source_path,
                sha256,
                speaker,
                content,
                emotion,
                sample_rate,
                duration_seconds,
                feature_shape,
                vector,
                spectrum,
                phase,
                energy,
                confirmed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                datetime.now(
                    timezone.utc
                ).isoformat(),

                str(audio_path),
                digest,
                speaker,
                content,
                emotion,
                code["sample_rate"],
                code["duration_seconds"],
                json.dumps(
                    code["feature_shape"]
                ),
                array_blob(
                    code["vector"]
                ),
                array_blob(
                    code["spectrum"]
                ),
                array_blob(
                    code["phase"]
                ),
                array_blob(
                    code["energy"]
                ),
            ),
        )

        connection.commit()

        print(
            json.dumps(
                {
                    "status": "stored",
                    "database": str(db_path),
                    "source": str(audio_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    except sqlite3.IntegrityError:
        print(
            json.dumps(
                {
                    "status": "already_stored",
                    "sha256": digest,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        connection.close()


def labelled_rows(
    connection: sqlite3.Connection,
    field: str,
) -> Iterable[sqlite3.Row]:
    allowed = {
        "speaker",
        "content",
        "emotion",
    }

    if field not in allowed:
        raise ValueError(
            f"未知のラベル項目: {field}"
        )

    return connection.execute(
        f"""
        SELECT
            id,
            {field},
            vector,
            source_path
        FROM samples
        WHERE confirmed = 1
          AND {field} IS NOT NULL
        """
    ).fetchall()


def recognize(
    db_path: Path,
    audio_path: Path,
    top: int,
) -> None:
    query_code = extract_code(audio_path)
    query_vector = query_code["vector"]

    connection = connect(db_path)

    result = {
        "status": "ok",
        "source": str(audio_path),
        "note": (
            "similarityは確率ではなく、"
            "保存例との近さです"
        ),
        "predictions": {},
    }

    try:
        for field in (
            "content",
            "speaker",
            "emotion",
        ):
            candidates = []

            for row in labelled_rows(
                connection,
                field,
            ):
                vector = blob_array(
                    row["vector"]
                ).astype(np.float32)

                similarity = float(
                    np.dot(
                        query_vector,
                        vector,
                    )
                )

                candidates.append(
                    {
                        "label": row[field],
                        "similarity": round(
                            similarity,
                            6,
                        ),
                        "sample_id": row["id"],
                        "source": row["source_path"],
                    }
                )

            candidates.sort(
                key=lambda item: item["similarity"],
                reverse=True,
            )

            result["predictions"][field] = (
                candidates[:max(1, top)]
            )

    finally:
        connection.close()

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


def export_code(
    audio_path: Path,
    output_path: Path,
) -> None:
    code = extract_code(audio_path)

    metadata = {
        key: value
        for key, value in code.items()
        if not isinstance(value, np.ndarray)
    }

    np.savez_compressed(
        output_path,
        vector=code["vector"],
        spectrum=code["spectrum"],
        phase=code["phase"],
        energy=code["energy"],
    )

    metadata["npz_file"] = str(output_path)

    print(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "音声スペクトラム・"
            "キャラクターコード実験"
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    add = sub.add_parser(
        "add",
        help=(
            "確認済みの音声例を"
            "経験データベースへ追加"
        ),
    )

    add.add_argument(
        "audio",
        type=Path,
    )

    add.add_argument(
        "--db",
        type=Path,
        default=Path(
            "audio_experience.sqlite3"
        ),
    )

    add.add_argument(
        "--speaker",
        help="話者ラベル",
    )

    add.add_argument(
        "--content",
        help="発音内容・文字・単語のラベル",
    )

    add.add_argument(
        "--emotion",
        help="感情表現ラベル",
    )

    recognize_parser = sub.add_parser(
        "recognize",
        help="保存例と比較して認識",
    )

    recognize_parser.add_argument(
        "audio",
        type=Path,
    )

    recognize_parser.add_argument(
        "--db",
        type=Path,
        default=Path(
            "audio_experience.sqlite3"
        ),
    )

    recognize_parser.add_argument(
        "--top",
        type=int,
        default=3,
    )

    export = sub.add_parser(
        "export",
        help="音声コードをNPZとして保存",
    )

    export.add_argument(
        "audio",
        type=Path,
    )

    export.add_argument(
        "output",
        type=Path,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "add":
        if not any(
            (
                args.speaker,
                args.content,
                args.emotion,
            )
        ):
            raise SystemExit(
                "--speaker、--content、--emotionのうち"
                "少なくとも一つを指定してください"
            )

        add_sample(
            args.db,
            args.audio,
            args.speaker,
            args.content,
            args.emotion,
        )

    elif args.command == "recognize":
        recognize(
            args.db,
            args.audio,
            args.top,
        )

    elif args.command == "export":
        export_code(
            args.audio,
            args.output,
        )


if __name__ == "__main__":
    main()
