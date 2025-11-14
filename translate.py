"""Standalone DeepSeek translation helper."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("未安装 openai，请先执行 `pip install openai`.") from exc


DEFAULT_PROMPT = (
    "你是一个专业的轻小说翻译家，擅长将日文轻小说内容自然、流畅地翻译成中文。"
    "忠实还原原文语气、情感和细节，避免直译。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 ReazonSpeech 文本进行批量翻译")
    parser.add_argument("input", help="原始 txt 文件路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 txt 路径，默认与输入同名 + 翻译.txt",
    )
    parser.add_argument("--api-key", default=None, help="DeepSeek/OpenAI API key")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI 兼容 API base URL，默认 https://api.deepseek.com",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="自定义系统提示词",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="单次请求的最大字符数",
    )
    return parser.parse_args()


def ensure_client(api_key: str | None, base_url: str | None) -> OpenAI:
    final_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not final_key:
        raise RuntimeError("未提供 API key，可使用 --api-key 或设置环境变量。")
    base = base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    return OpenAI(api_key=final_key, base_url=base)


def split_into_chunks(text: str, max_chars: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for line in text.splitlines(keepends=True):
        line_len = len(line)
        if current and current_len + line_len > max_chars:
            chunks.append("".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len

    if current:
        chunks.append("".join(current))

    return chunks


def translate_chunk(
    client: OpenAI, prompt: str, chunk_text: str, idx: int, total: int
) -> str:
    print(f"正在翻译第 {idx}/{total} 段，长度 {len(chunk_text)} 字符...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"这是第 {idx}/{total} 段内容，请直接输出对应的中文翻译：\n\n{chunk_text}"
                ),
            },
        ],
        stream=True,
    )

    translated = ""
    for chunk in response:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            translated += delta.content

    return translated.strip()


def translate_file(
    client: OpenAI,
    input_text: str,
    prompt: str,
    max_chars: int,
) -> List[str]:
    chunks = split_into_chunks(input_text, max_chars)
    total = len(chunks)
    if total == 0:
        raise ValueError("未在源文件中读取到内容。")
    return [
        translate_chunk(client, prompt, chunk, idx, total)
        for idx, chunk in enumerate(chunks, 1)
    ]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    output_path = (
        args.output
        if args.output is not None
        else Path(os.path.splitext(str(input_path))[0] + "翻译.txt")
    )

    client = ensure_client(args.api_key, args.base_url)
    input_text = input_path.read_text(encoding="utf-8")
    translations = translate_file(client, input_text, args.prompt, args.max_chars)
    content = "\n".join(translations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    print("\n--- 翻译结果已保存 ---\n")
    print(output_path)


if __name__ == "__main__":
    main()
