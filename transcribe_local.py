#!/usr/bin/env python3
"""
Transcreve áudio local OU vídeo do YouTube para .txt usando faster-whisper.

Uso:
    python transcribe_local.py audio.ogg
    python transcribe_local.py https://www.youtube.com/watch?v=XXXX
    python transcribe_local.py URL_OU_ARQUIVO -o saida.txt

Env vars:
    FASTER_WHISPER_MODEL   tiny | base | small | medium | large-v3   (default: small)
    FASTER_WHISPER_DEVICE  auto | cpu | cuda                          (default: auto)
    FASTER_WHISPER_COMPUTE int8 | float16 | float32                   (default: int8)
"""
import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

URL_RE = re.compile(r"^https?://", re.I)


def is_url(s: str) -> bool:
    return bool(URL_RE.match(s))


def download_audio(url: str, out_dir: Path) -> tuple[Path, str]:
    """Baixa o áudio do YouTube (ou qualquer site suportado por yt-dlp) como .m4a."""
    import yt_dlp

    out_template = str(out_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "0"}
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title") or info.get("id") or "audio"
        audio_path = out_dir / f"{info['id']}.m4a"
        if not audio_path.exists():
            # fallback: pega o primeiro arquivo gerado
            audio_path = next(out_dir.iterdir())
        return audio_path, title


def transcribe(audio_path: Path) -> dict:
    model_size = os.getenv("FASTER_WHISPER_MODEL", "small")
    device = os.getenv("FASTER_WHISPER_DEVICE", "auto")
    compute = os.getenv("FASTER_WHISPER_COMPUTE", "int8")

    model = WhisperModel(model_size, device=device, compute_type=compute)
    segments, info = model.transcribe(str(audio_path), vad_filter=True)

    segs = []
    for s in segments:
        segs.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        print(f"[{s.start:7.2f} -> {s.end:7.2f}] {s.text.strip()}", file=sys.stderr)

    return {
        "text": " ".join(s["text"] for s in segs),
        "segments": segs,
        "language": info.language,
        "model": model_size,
        "device": device,
        "compute": compute,
    }


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-. ]+", "_", name).strip()[:120] or "transcricao"


def main() -> int:
    p = argparse.ArgumentParser(description="Transcreve áudio local ou vídeo do YouTube para .txt")
    p.add_argument("source", help="Caminho de arquivo de áudio/vídeo OU URL (YouTube etc.)")
    p.add_argument("-o", "--output", help="Arquivo .txt de saída (default: <titulo>.txt)")
    p.add_argument("--json", action="store_true", help="Também emite JSON com segmentos")
    args = p.parse_args()

    tmpdir = None
    try:
        if is_url(args.source):
            tmpdir = tempfile.mkdtemp(prefix="ytdlp_")
            print(f"Baixando áudio de: {args.source}", file=sys.stderr)
            audio_path, title = download_audio(args.source, Path(tmpdir))
            print(f"Áudio salvo em: {audio_path}", file=sys.stderr)
        else:
            audio_path = Path(args.source)
            if not audio_path.exists():
                print(f"Arquivo não encontrado: {audio_path}", file=sys.stderr)
                return 1
            title = audio_path.stem

        out_txt = Path(args.output) if args.output else Path(f"{safe_filename(title)}.txt")
        print(f"Transcrevendo... (modelo={os.getenv('FASTER_WHISPER_MODEL', 'small')})", file=sys.stderr)
        result = transcribe(audio_path)

        out_txt.write_text(result["text"].strip() + "\n", encoding="utf-8")
        print(f"\nOK: transcrição salva em {out_txt} ({len(result['text'])} chars, idioma={result['language']})", file=sys.stderr)

        if args.json:
            out_json = out_txt.with_suffix(".json")
            out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"OK: JSON salvo em {out_json}", file=sys.stderr)

        return 0
    finally:
        if tmpdir:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
