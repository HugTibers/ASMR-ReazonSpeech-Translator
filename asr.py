"""Minimal CLI for running a single ReazonSpeech ASR job."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import torch
from espnet2.bin.asr_inference import Speech2Text
from reazonspeech.espnet.asr import audio_from_path, transcribe

DEFAULT_MODEL_SOURCE = "https://huggingface.co/reazon-research/reazonspeech-espnet-v2"
MODEL_SOURCE_ENV = "REAZONSPEECH_MODEL_SOURCE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio with ReazonSpeech")
    parser.add_argument("audio", help="待转写的音频路径")
    parser.add_argument(
        "--device",
        default="cuda",
        help="载入模型所用的设备，例如 cuda / cpu",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="结果保存路径，不指定则与音频同名 (_reazon.txt)",
    )
    parser.add_argument(
        "--model-source",
        default=None,
        help=(
            "ReazonSpeech 模型来源，可为本地目录、压缩包或 HuggingFace URL；"
            f"默认使用 {DEFAULT_MODEL_SOURCE}，也可通过环境变量 {MODEL_SOURCE_ENV} 覆盖。"
        ),
    )
    return parser.parse_args()


def resolve_model_source(explicit: str | None) -> str:
    return explicit or os.getenv(MODEL_SOURCE_ENV) or DEFAULT_MODEL_SOURCE


def load_reazonspeech_model(device: str, source: str | None) -> Speech2Text:
    final_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_source = resolve_model_source(source)
    print(f"[ASR] 正在从 {model_source} 加载 ReazonSpeech 模型（设备：{final_device}）")
    return Speech2Text.from_pretrained(model_source, lm_weight=0, device=final_device)


def run_asr(
    audio_file: Path, device: str, model_source: str | None
) -> Iterable[Tuple[float, float, str]]:
    model = load_reazonspeech_model(device, model_source)
    audio = audio_from_path(str(audio_file))
    result = transcribe(model, audio)
    for segment in result.segments:
        start = float(getattr(segment, "start_seconds", 0.0))
        end = float(getattr(segment, "end_seconds", 0.0))
        text = getattr(segment, "text", str(segment))
        yield start, end, text


def save_segments(segments: Iterable[Tuple[float, float, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for start, end, text in segments:
            line = f"[{start:.2f}s -> {end:.2f}s] {text}"
            print("  -", line)
            fout.write(line + "\n")


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")

    output_path = (
        args.output
        if args.output is not None
        else Path(os.path.splitext(str(audio_path))[0] + "_reazon.txt")
    )

    segments = list(run_asr(audio_path, args.device, args.model_source))
    save_segments(segments, output_path)
    print(f"分段结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
    
