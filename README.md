# ReazonSpeech + DeepSeek 一键字幕流水线

本仓库基于 ReazonSpeech 的 ESPnet 模型完成语音识别，并使用 DeepSeek (OpenAI 兼容接口) 进行翻译，再通过 ffmpeg 生成中日双语字幕视频。
## 效果图
- 日语识别
![alt text](test/日语识别.png)
- 日语翻译
![alt text](test/日语翻译.png)

## 目录

- [ReazonSpeech + DeepSeek 一键字幕流水线](#reazonspeech--deepseek-一键字幕流水线)
  - [效果图](#效果图)
  - [目录](#目录)
  - [环境准备](#环境准备)
  - [单次 ASR：`asr.py`](#单次-asrasrpy)
  - [仅翻译文本：`translate.py`](#仅翻译文本translatepy)
  - [单独生成 SRT：`srt.py`](#单独生成-srtsrtpy)
  - [完整流水线：`pipeline.py`](#完整流水线pipelinepy)
  - [批量处理文件夹：`batch_pipeline.py`](#批量处理文件夹batch_pipelinepy)
  - [常见问题](#常见问题)
  - [语音识别](#语音识别)

---

## 环境准备

```bash
# 1. 下载项目
git clone https://github.com/HugTibers/ASMR-ReazonSpeech-Translator.git
# 2. 获取 ReazonSpeech 代码与依赖
pip install "ReazonSpeech/pkg/espnet-asr"

// nemo-asr对配置要求较高

# 3. 安装本项目所需依赖
pip install huggingface-hub openai soundfile numpy==1.26.4

# 4. 确保本机已安装 ffmpeg，并且在 PATH 中, 不需要合成视频就不用下载
ffmpeg -version

Ubuntu: sudo yum install ffmpeg
Windows: 访问 ffmpeg官网 或 Gyan.dev下载页，下载 Windows 版本压缩包。
```

> **提示**：DeepSeek 相关接口与 OpenAI SDK 兼容，脚本默认通过 `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` 注入密钥，你也可以在命令中使用 `--api-key` 直接传入。

---

## 单次 ASR：`asr.py`

`asr.py` 是最小可复用的命令行工具，用于把单个音频转写为带时间戳的 TXT。

```bash
python asr.py test/Track1.wav
```

- 默认输出为 `音频名_reazon.txt`。
- `--device` 支持 `cuda` / `cpu`；无 GPU 时可切换为 `--device cpu`。

---

## 仅翻译文本：`translate.py`

当你已有 ReazonSpeech 的转写文本时，可单独调用 DeepSeek 翻译。

```bash
export DEEPSEEK_API_KEY=sk-xxx  # 或 OPENAI_API_KEY

python translate.py test/Track1_reazon.txt --api-key sk-xxx
  
```
---

## 单独生成 SRT：`srt.py`

如果只想把现有的中/日 TXT 合并成字幕：

```bash
python srt.py \
  test/Track1_reazon.txt \
  test/Track1_reazon翻译.txt
```

当行数不匹配时脚本会给出警告，多余的句子会仅显示日文。

---

## 完整流水线：`pipeline.py`

`pipeline.py` 串联了 ASR → 翻译 → SRT → ffmpeg 视频四个阶段，并带有断点续跑逻辑。基础用法：

```bash
python pipeline.py test/Track1.wav \
  --api-key $DEEPSEEK_API_KEY
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--device` | ReazonSpeech 运行设备，默认 `cuda` |
| `--api-key` / `--base-url` | DeepSeek/OpenAI 兼容接口配置 |
| `--prompt` / `--prompt-file` | 覆盖默认翻译提示词，`--prompt-file` 优先级更高 |
| `--max-chars` | 单次翻译请求的最大字符数 (默认 4000) |
| `--bg-color` / `--resolution` | ffmpeg 生成字幕视频的背景色和分辨率 |
| `--no-resume` | 关闭断点续跑，强制从头执行 |

执行过程中会生成如下文件：

- `test/Track1_reazon.txt`：日文转写结果
- `test/Track1_reazon翻译.txt`：中文翻译（保持时间戳）
- `test/Track1_reazon.srt`：中日双语字幕
- `test/Track1_subbed.mp4`：带字幕的视频
- `test/Track1_pipeline_state.json`：断点续跑状态（成功结束会自动删除）

---

## 批量处理文件夹：`batch_pipeline.py`

对多个音频重复执行 `pipeline.py`：

```bash
// test是文件夹
python batch_pipeline.py test \
  --extensions .mp3 .wav \
  --pipeline-args --api-key <key>
```

示例：

```bash
python batch_pipeline.py "test" --pipeline-args --api-key sk-qweqweqwe123123131 (使用你自己的deepseek秘钥)
```

- 默认会跳过已存在 `*_subbed.mp4` 的文件。
- `--pipeline-script` 可指向自定义的 pipeline 路径。
- 通过 `--pipeline-args` 把其他参数透传给 `pipeline.py`。

---

## 常见问题

1. **无法下载模型**：请确保服务器能够访问 Hugging Face（必要时配置代理或镜像），脚本会在运行时自动拉取官方模型。
2. **无 GPU 环境**：所有脚本都可通过 `--device cpu` 运行，只是速度会慢。
3. **ffmpeg 报错**：请确认已安装 ffmpeg 5.x+ 且路径无中文或空格；字幕路径中若包含特殊字符已自动转义。


## 语音识别
普通对话可以正常识别，但是哦吼哦吼的叫声基本不能识别，不过这种声音大多数不用翻译也能明白。