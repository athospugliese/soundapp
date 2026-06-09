"""
Diarization local via NeMo (NVIDIA), sem dependência do HuggingFace.

Estratégia:
  1. Converte o áudio pra mono 16kHz (requisito do NeMo).
  2. Baixa o YAML canônico de inferência do NeMo (cacheado em ~/.cache/nemo_msdd_configs).
  3. Roda NeMo MSDD (Multi-Scale Diarization Decoder) via NeuralDiarizer.
  4. Recebe um RTTM com [start, dur, speaker_label].
  5. Mescla com os segmentos do faster-whisper por sobreposição temporal.

Modelos baixam automaticamente do NGC da NVIDIA na primeira execução (~600MB).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CONFIG_URLS = {
    "general": "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_general.yaml",
    "telephonic": "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml",
    "meeting": "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_meeting.yaml",
}

CONFIG_CACHE_DIR = Path.home() / ".cache" / "nemo_msdd_configs"

# Modelos NeMo locais (movidos do cache para o projeto). Se existirem, são usados
# offline; caso contrário, o NeMo baixa do NGC na primeira execução.
NEMO_DIR = Path(__file__).parent / "models" / "nemo"
LOCAL_VAD = NEMO_DIR / "vad_multilingual_marblenet.nemo"
LOCAL_SPEAKER = NEMO_DIR / "titanet-l.nemo"


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    text: str


def _to_mono16k(audio_path: Path, out_dir: Path) -> Path:
    """Converte qualquer áudio pra mono 16kHz WAV (necessário pro NeMo)."""
    out = out_dir / "audio_mono16k.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(out),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out


def _build_manifest(wav_path: Path, out_dir: Path, num_speakers: int | None) -> Path:
    """Cria o manifesto JSONL que o NeMo espera."""
    manifest = out_dir / "input_manifest.json"
    entry = {
        "audio_filepath": str(wav_path),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": num_speakers,
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return manifest


def _get_canonical_config(preset: str = "general") -> Path:
    """Baixa (cacheado) o YAML de inferência canônico do NeMo."""
    local_cfg = NEMO_DIR / f"diar_infer_{preset}.yaml"
    if local_cfg.exists():
        return local_cfg
    CONFIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CONFIG_CACHE_DIR / f"diar_infer_{preset}.yaml"
    if not cfg_path.exists():
        url = CONFIG_URLS[preset]
        urllib.request.urlretrieve(url, cfg_path)
    return cfg_path


def _diarize_with_nemo(
    wav_path: Path,
    out_dir: Path,
    num_speakers: int | None,
    preset: str = "general",
) -> list[tuple[float, float, str]]:
    """Roda o NeMo MSDD e retorna lista de (start, end, speaker_label)."""
    from omegaconf import OmegaConf
    from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer

    cfg_yaml = _get_canonical_config(preset)
    cfg = OmegaConf.load(cfg_yaml)

    manifest = _build_manifest(wav_path, out_dir, num_speakers)

    # Sobrescrevemos só o que precisamos — o YAML canônico já tem todos os campos obrigatórios.
    OmegaConf.set_struct(cfg, False)
    cfg.num_workers = 0
    cfg.device = "cpu"
    # Usa os .nemo locais do projeto se disponíveis (evita download do NGC).
    if LOCAL_VAD.exists():
        cfg.diarizer.vad.model_path = str(LOCAL_VAD)
    if LOCAL_SPEAKER.exists():
        cfg.diarizer.speaker_embeddings.model_path = str(LOCAL_SPEAKER)
    cfg.diarizer.manifest_filepath = str(manifest)
    cfg.diarizer.out_dir = str(out_dir)
    cfg.diarizer.oracle_vad = False
    cfg.diarizer.collar = 0.25
    cfg.diarizer.ignore_overlap = True
    cfg.diarizer.clustering.parameters.oracle_num_speakers = num_speakers is not None
    if num_speakers is not None:
        cfg.diarizer.clustering.parameters.max_num_speakers = num_speakers

    diarizer = ClusteringDiarizer(cfg=cfg)
    diarizer.diarize()

    rttm_path = out_dir / "pred_rttms" / (wav_path.stem + ".rttm")
    if not rttm_path.exists():
        raise RuntimeError(f"RTTM não gerado em {rttm_path}")

    segments: list[tuple[float, float, str]] = []
    for line in rttm_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        dur = float(parts[4])
        speaker = parts[7]
        segments.append((start, start + dur, speaker))
    return segments


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge_whisper_with_speakers(
    whisper_segments: list[dict],
    speaker_segments: list[tuple[float, float, str]],
) -> list[SpeakerSegment]:
    """Para cada segmento do Whisper, atribui o speaker com maior sobreposição."""
    out: list[SpeakerSegment] = []
    for w in whisper_segments:
        w_start, w_end, w_text = w["start"], w["end"], w["text"]
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for s_start, s_end, s_spk in speaker_segments:
            ov = _overlap(w_start, w_end, s_start, s_end)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = s_spk
        out.append(SpeakerSegment(start=w_start, end=w_end, speaker=best_speaker, text=w_text))
    return out


def diarize_and_merge(
    audio_path: Path,
    whisper_segments: list[dict],
    num_speakers: int | None = None,
    preset: str = "general",
) -> list[SpeakerSegment]:
    """Pipeline completo: prepara áudio → diariza com NeMo → mescla com Whisper."""
    with tempfile.TemporaryDirectory(prefix="nemo_diar_") as tmp:
        tmp_path = Path(tmp)
        wav = _to_mono16k(audio_path, tmp_path)
        speaker_segs = _diarize_with_nemo(wav, tmp_path, num_speakers, preset)
    return merge_whisper_with_speakers(whisper_segments, speaker_segs)


def format_as_dialogue(segments: list[SpeakerSegment]) -> str:
    """Agrupa turnos consecutivos do mesmo falante."""
    if not segments:
        return ""
    lines = []
    current_spk = segments[0].speaker
    current_text = [segments[0].text]
    current_start = segments[0].start
    for s in segments[1:]:
        if s.speaker == current_spk:
            current_text.append(s.text)
        else:
            mm, ss = divmod(int(current_start), 60)
            lines.append(f"[{mm:02d}:{ss:02d}] {current_spk}: {' '.join(current_text).strip()}")
            current_spk = s.speaker
            current_text = [s.text]
            current_start = s.start
    mm, ss = divmod(int(current_start), 60)
    lines.append(f"[{mm:02d}:{ss:02d}] {current_spk}: {' '.join(current_text).strip()}")
    return "\n\n".join(lines)
