import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

"""
Usage:
    python pipeline.py --audio "./target/sample.mp3"
"""

from reazonspeech.espnet.asr import audio_from_path, load_model, transcribe

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


TIMED_LINE = re.compile(r"\[(?P<start>[\d.]+)s -> (?P<end>[\d.]+)s\] (?P<text>.*)")
STAGES: Tuple[str, ...] = ("asr", "translate", "srt", "video")
PROGRESS_SUFFIX = "_pipeline_state.json"
DEFAULT_PROMPT = (
    "你是一个专业的轻小说翻译家，擅长将日文轻小说内容自然、流畅地翻译成中文。"
    "忠实还原原文语气、情感和细节，避免直译，确保输出仍保留原文的分行与时间标记。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ReazonSpeech 识别 + DeepSeek 翻译 + SRT + ffmpeg 一键流水线"
    )
    parser.add_argument("audio", help="需要处理的音频文件路径")
    parser.add_argument(
        "--device",
        default="cuda",
        help="加载 ReazonSpeech 模型所用设备 (cuda / cpu)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="DeepSeek / OpenAI 兼容 API key，不填则读取环境变量",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="可选的 OpenAI 兼容接口 Base URL，默认使用 https://api.deepseek.com",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="覆盖默认翻译提示词",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="从文件读取提示词，优先级高于 --prompt",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="翻译时单次请求允许的最大字符数",
    )
    parser.add_argument(
        "--bg-color",
        default="black",
        help="ffmpeg 生成视频时的背景颜色",
    )
    parser.add_argument(
        "--resolution",
        default="1280x720",
        help="输出视频分辨率",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用断点续跑逻辑，总是从头开始执行",
    )
    return parser.parse_args()


def build_artifact_path(audio_path: Path, suffix: str) -> Path:
    base = Path(os.path.splitext(str(audio_path))[0])
    return Path(str(base) + suffix)


def build_progress_path(audio_path: Path) -> Path:
    return build_artifact_path(audio_path, PROGRESS_SUFFIX)


def load_progress_state(progress_path: Path) -> str | None:
    if not progress_path.exists():
        return None
    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    stage = data.get("last_completed")
    return stage if stage in STAGES else None


def save_progress_state(progress_path: Path, stage: str | None) -> None:
    payload = {"last_completed": stage}
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_transcript(segments: Sequence[Tuple[float, float, str]], txt_path: Path) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as fout:
        for start, end, text in segments:
            fout.write(f"[{start:.2f}s -> {end:.2f}s] {text}\n")


def run_reazonspeech(audio_path: Path, device: str) -> List[Tuple[float, float, str]]:
    model = load_model(device)
    audio = audio_from_path(str(audio_path))
    result = transcribe(model, audio)
    segments: List[Tuple[float, float, str]] = []
    for segment in result.segments:
        start = float(getattr(segment, "start_seconds", 0.0))
        end = float(getattr(segment, "end_seconds", 0.0))
        text = getattr(segment, "text", str(segment))
        segments.append((start, end, text))
    return segments


def split_chunks(text: str, max_chars: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for line in text.splitlines(keepends=True):
        line_len = len(line)
        if current and length + line_len > max_chars:
            chunks.append("".join(current))
            current = [line]
            length = line_len
        else:
            current.append(line)
            length += line_len
    if current:
        chunks.append("".join(current))
    return chunks


def ensure_openai_client(api_key: str | None, base_url: str | None) -> OpenAI:
    if OpenAI is None:  # pragma: no cover
        raise ModuleNotFoundError("未安装 openai，请先执行 `pip install openai`。")
    final_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not final_key:
        raise RuntimeError("未提供 DeepSeek/OpenAI API key，可使用 --api-key 或环境变量。")
    base = base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    return OpenAI(api_key=final_key, base_url=base)


def translate_text(
    client: OpenAI, prompt: str, text: str, max_chars: int
) -> str:
    chunks = split_chunks(text, max_chars)
    total = len(chunks)
    translations: List[str] = []
    for idx, chunk in enumerate(chunks, 1):
        print(f"[DeepSeek] 正在翻译段落 {idx}/{total}，长度 {len(chunk)}")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"以下是整体的第 {idx}/{total} 段内容，请保持时间戳格式并翻译为中文：\n\n{chunk}"
                    ),
                },
            ],
            stream=True,
        )
        translated = ""
        for piece in response:
            delta = piece.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                translated += delta.content
        translations.append(translated.strip())
    return "\n".join(translations).strip() + "\n"


def parse_timed_file(path: Path) -> List[Tuple[float, float, str]]:
    entries: List[Tuple[float, float, str]] = []
    with path.open("r", encoding="utf-8") as fin:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            match = TIMED_LINE.match(raw)
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


def build_srt(
    jp_entries: Sequence[Tuple[float, float, str]],
    cn_entries: Sequence[Tuple[float, float, str]],
    srt_path: Path,
) -> None:
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    with srt_path.open("w", encoding="utf-8") as fout:
        for idx, entry in enumerate(jp_entries, 1):
            start, end, text = entry
            cn_text = cn_entries[idx - 1][2] if idx - 1 < len(cn_entries) else ""
            body = text if not cn_text else f"{text}\n{cn_text}"
            fout.write(f"{idx}\n{format_time(start)} --> {format_time(end)}\n{body}\n\n")
    if len(jp_entries) != len(cn_entries):
        print("警告：翻译行数不足，未匹配的句子将只显示日文。")


def escape_for_subtitles(path: Path) -> str:
    # Escape characters that ffmpeg subtitles filter treats specially.
    escaped = str(path)
    for src, repl in [
        ("\\", "\\\\"),
        (":", "\\:"),
        ("'", "\\'"),
        (",", "\\,"),
        ("[", "\\["),
        ("]", "\\]"),
        (" ", "\\ "),
    ]:
        escaped = escaped.replace(src, repl)
    return escaped



import mimetypes
import soundfile as sf

def get_audio_codec(audio_path: Path) -> str:
    # 只对 aac、mp3、mpeg 直接 copy，其他一律转码为 aac，避免 mp4 容器不兼容
    mime, _ = mimetypes.guess_type(str(audio_path))
    if mime:
        if 'mp3' in mime or 'aac' in mime or 'mpeg' in mime:
            return 'copy'
    # 其他情况一律转码为 aac
    return 'aac'

def mux_video(
    audio_path: Path,
    srt_path: Path,
    duration: float,
    output_path: Path,
    bg_color: str,
    resolution: str,
) -> None:
    escaped_sub = escape_for_subtitles(srt_path)
    lavfi = f"color=c={bg_color}:s={resolution}:d={max(duration, 1):.3f}"
    audio_codec = get_audio_codec(audio_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        lavfi,
        "-i",
        str(audio_path),
        "-vf",
        f"subtitles={escaped_sub}",
        "-c:v",
        "libx264",
        "-c:a",
        audio_codec,
        "-shortest",
        str(output_path),
    ]
    print(f"[ffmpeg] 正在生成带字幕视频...（音频编码: {audio_codec}）")
    subprocess.run(cmd, check=True)


def resolve_prompt(prompt_arg: str | None, prompt_file: Path | None) -> str:
    if prompt_file:
        return prompt_file.expanduser().read_text(encoding="utf-8")
    if prompt_arg:
        return prompt_arg
    return DEFAULT_PROMPT


def main() -> None:
    args = parse_args()
    resume_enabled = not args.no_resume
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")

    txt_path = build_artifact_path(audio_path, "_reazon.txt")
    trans_path = build_artifact_path(audio_path, "_reazon翻译.txt")
    srt_path = build_artifact_path(audio_path, "_reazon.srt")
    output_path = audio_path.with_name(audio_path.stem + "_subbed.mp4")

    artifact_map = {
        "asr": txt_path,
        "translate": trans_path,
        "srt": srt_path,
        "video": output_path,
    }
    stage_desc = {
        "asr": "转写结果",
        "translate": "翻译结果",
        "srt": "字幕文件",
        "video": "字幕视频",
    }

    progress_path = build_progress_path(audio_path)
    start_idx = 0
    if resume_enabled:
        last_completed = load_progress_state(progress_path)
        if last_completed:
            print(f"[Resume] 上次已完成阶段：{last_completed}")
            start_idx = min(STAGES.index(last_completed) + 1, len(STAGES))
        for idx in range(start_idx):
            stage = STAGES[idx]
            if not artifact_map[stage].exists():
                print(
                    f"[Resume] 找不到 {stage_desc[stage]} {artifact_map[stage]}, 从 {stage} 阶段重新执行"
                )
                start_idx = idx
                break
        if start_idx >= len(STAGES) and artifact_map["video"].exists():
            print("[Resume] 检测到所有阶段已完成，如需重新执行请删除进度文件或使用 --no-resume")
            return
    else:
        if progress_path.exists():
            progress_path.unlink()

    client = None
    prompt = resolve_prompt(args.prompt, args.prompt_file)
    for idx, stage in enumerate(STAGES):
        if idx < start_idx:
            continue
        try:
            if stage == "asr":
                segments = run_reazonspeech(audio_path, args.device)
                write_transcript(segments, txt_path)
                print(f"[ASR] 转写完成：{txt_path}")
            elif stage == "translate":
                if client is None:
                    client = ensure_openai_client(args.api_key, args.base_url)
                with txt_path.open("r", encoding="utf-8") as fin:
                    source_text = fin.read()
                translation = translate_text(client, prompt, source_text, args.max_chars)
                trans_path.parent.mkdir(parents=True, exist_ok=True)
                trans_path.write_text(translation, encoding="utf-8")
                print(f"[DeepSeek] 翻译完成：{trans_path}")
            elif stage == "srt":
                jp_entries = parse_timed_file(txt_path)
                cn_entries = parse_timed_file(trans_path)
                build_srt(jp_entries, cn_entries, srt_path)
                print(f"[SRT] 已生成字幕文件：{srt_path}")
            elif stage == "video":
                jp_entries = parse_timed_file(txt_path)
                duration = max((end for _, end, _ in jp_entries), default=0.0)
                mux_video(
                    audio_path,
                    srt_path,
                    duration,
                    output_path,
                    args.bg_color,
                    args.resolution,
                )
                print(f"[完成] 已输出视频：{output_path}")
        except Exception:
            print(f"[Resume] 阶段 {stage} 失败，下次运行将从该阶段继续。")
            raise
        else:
            if resume_enabled:
                save_progress_state(progress_path, stage)

    if resume_enabled and progress_path.exists():
        progress_path.unlink()


if __name__ == "__main__":
    main()
