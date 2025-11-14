from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple


TIMED_LINE = re.compile(r"\[(?P<start>[\d.]+)s -> (?P<end>[\d.]+)s\] (?P<text>.*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 ReazonSpeech 文本组合成 SRT")
    parser.add_argument("jp", help="原始转写 txt 路径")
    parser.add_argument("cn", help="翻译 txt 路径")
    parser.add_argument("output", help="输出的 srt 路径")
    return parser.parse_args()


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def parse_lines(path: Path) -> List[Tuple[float, float, str]]:
    entries: List[Tuple[float, float, str]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            match = TIMED_LINE.match(line.strip())
            if not match:
                continue
            entries.append(
                (
                    float(match.group("start")),
                    float(match.group("end")),
                    match.group("text"),
                )
            )
    return entries


def write_srt(
    jp_entries: Iterable[Tuple[float, float, str]],
    cn_entries: List[Tuple[float, float, str]],
    srt_path: Path,
) -> None:
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    with srt_path.open("w", encoding="utf-8") as fout:
        for idx, entry in enumerate(jp_entries, 1):
            start, end, text = entry
            cn_text = cn_entries[idx - 1][2] if idx - 1 < len(cn_entries) else ""
            subtitle_text = text if not cn_text else f"{text}\n{cn_text}"
            fout.write(
                f"{idx}\n{format_time(start)} --> {format_time(end)}\n{subtitle_text}\n\n"
            )


def main() -> None:
    args = parse_args()
    jp_path = Path(args.jp).expanduser().resolve()
    cn_path = Path(args.cn).expanduser().resolve()
    srt_path = Path(args.output).expanduser().resolve()

    original_entries = parse_lines(jp_path)
    cn_entries = parse_lines(cn_path)

    if len(cn_entries) != len(original_entries):
        print("警告：中日行数不匹配，缺失的翻译将留空。")

    write_srt(original_entries, cn_entries, srt_path)
    print(f"SRT 字幕已生成：{srt_path}")


if __name__ == "__main__":
    main()
