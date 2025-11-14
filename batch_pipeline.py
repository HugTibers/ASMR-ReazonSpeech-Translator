from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".aac")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对文件夹内的音频批量执行 pipeline.py")
    parser.add_argument("folder", help="音频文件所在的文件夹路径")
    parser.add_argument(
        "--pipeline-script",
        default="pipeline.py",
        help="要调用的 pipeline 脚本，默认与本文件同级的 pipeline.py",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="需要处理的音频扩展名 (含点)，默认常见音频",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="单个文件失败后的最大重试次数",
    )
    parser.add_argument(
        "--pipeline-args",
        nargs=argparse.REMAINDER,
        default=(),
        help="传给 pipeline.py 的附加参数，例如 --pipeline-args --no-resume",
    )
    return parser.parse_args()


def resolve_pipeline_script(candidate: str) -> Path:
    path = Path(candidate)
    if path.exists():
        return path.resolve()
    alt = Path(__file__).with_name(candidate)
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(f"找不到 pipeline 脚本：{candidate}")


def discover_audio_files(folder: Path, extensions: Sequence[str]) -> list[Path]:
    wanted = {ext.lower() for ext in extensions}
    return sorted(
        f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in wanted
    )


def batch_translate(
    folder: str,
    pipeline_script: str,
    extensions: Iterable[str],
    max_retries: int,
    pipeline_args: Sequence[str],
) -> None:
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        raise NotADirectoryError(f"文件夹不存在: {folder_path}")

    script_path = resolve_pipeline_script(pipeline_script)
    files = discover_audio_files(folder_path, extensions)
    if not files:
        print(f"未在 {folder_path} 下找到音频文件 (扩展: {', '.join(extensions)})")
        return

    for audio_file in files:
        mp4_file = audio_file.with_name(audio_file.stem + "_subbed.mp4")
        if mp4_file.exists():
            print(f"已存在 mp4，跳过: {mp4_file.name}")
            continue

        print(f"\n==== 处理: {audio_file.name} ====")
        for attempt in range(1, max_retries + 1):
            try:
                cmd = [sys.executable, str(script_path), str(audio_file), *pipeline_args]
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"第{attempt}次处理失败: {audio_file.name}，错误: {exc}")
                if attempt == max_retries:
                    print(f"已达最大重试次数，跳过: {audio_file.name}")
            else:
                print(f"成功: {audio_file.name}")
                break


def main() -> None:
    args = parse_args()
    batch_translate(
        folder=args.folder,
        pipeline_script=args.pipeline_script,
        extensions=args.extensions,
        max_retries=args.max_retries,
        pipeline_args=args.pipeline_args,
    )


if __name__ == "__main__":
    main()
