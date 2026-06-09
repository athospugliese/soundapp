"""
Streamlit UI para transcrição local com faster-whisper.
Aceita URL (YouTube etc.) ou upload de arquivo de áudio/vídeo.
"""
import os
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel

from transcribe_local import download_audio, is_url, safe_filename

try:
    from diarize import diarize_and_merge, format_as_dialogue
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False

st.set_page_config(page_title="Transcrição Local", page_icon="🎙️", layout="wide")

MODELS_DIR = Path(__file__).parent / "models"


def resolve_model(model_size: str) -> str:
    """Usa o modelo local em models/ se existir; senão devolve o nome p/ download."""
    local = MODELS_DIR / f"faster-whisper-{model_size}"
    if (local / "model.bin").exists():
        return str(local)
    return model_size


@st.cache_resource(show_spinner="Carregando modelo Whisper...")
def load_model(model_size: str, device: str, compute: str) -> WhisperModel:
    return WhisperModel(resolve_model(model_size), device=device, compute_type=compute)


def run_transcription(audio_path: Path, model: WhisperModel):
    segments_iter, info = model.transcribe(str(audio_path), vad_filter=True)
    segs = []
    duration = info.duration or 0

    st.markdown("### 🔴 Transcrição ao vivo")
    progress = st.progress(0.0, text=f"Iniciando... (idioma detectado: {info.language})")
    live_text = st.empty()
    last_seg = st.empty()

    accumulated = ""
    for s in segments_iter:
        text = s.text.strip()
        segs.append({"start": s.start, "end": s.end, "text": text})
        accumulated += (" " if accumulated else "") + text

        if duration:
            pct = min(s.end / duration, 1.0)
            progress.progress(
                pct,
                text=f"Transcrevendo... {s.end:6.1f}s / {duration:.1f}s  ({pct*100:.0f}%)",
            )
        live_text.text_area(
            "Texto acumulado",
            accumulated,
            height=300,
            label_visibility="collapsed",
            key=f"live_{len(segs)}",
        )
        last_seg.markdown(f"**↳ último segmento:** `[{s.start:.2f}s → {s.end:.2f}s]` _{text}_")

    progress.progress(1.0, text="✅ Concluído")
    return segs, info


st.title("🎙️ Transcrição Local — YouTube + Faster-Whisper")
st.caption("100% local. Sem token, sem nuvem. Rodando no seu Mac.")

with st.sidebar:
    st.header("⚙️ Configurações")
    model_size = st.selectbox(
        "Modelo Whisper",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=2,
        help="Maior = melhor qualidade, mais lento. `small` é um bom equilíbrio em CPU.",
    )
    device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    compute = st.selectbox(
        "Compute type",
        ["int8", "int8_float16", "float16", "float32"],
        index=0,
        help="`int8` é o mais rápido em CPU.",
    )
    st.divider()
    st.markdown("**🗣️ Diarization (identificar falantes)**")
    enable_diar = st.checkbox(
        "Ativar diarization (NeMo)",
        value=False,
        disabled=not DIARIZATION_AVAILABLE,
        help="Identifica quem fala. Usa NeMo da NVIDIA (sem HuggingFace). Mais lento.",
    )
    num_speakers_input = st.text_input(
        "Número de falantes (vazio = auto)",
        value="",
        disabled=not enable_diar,
        help="Se você sabe quantos falantes têm, especifique. Caso contrário, deixe vazio.",
    )
    try:
        num_speakers = int(num_speakers_input) if num_speakers_input.strip() else None
    except ValueError:
        num_speakers = None
        st.warning("Número inválido — usando auto-detect.")

    st.divider()
    st.markdown("**Status**")
    st.write(f"ffmpeg: `{'✅' if os.system('which ffmpeg > /dev/null') == 0 else '❌'}`")
    st.write(f"diarization: `{'✅ NeMo' if DIARIZATION_AVAILABLE else '⏳ instalando...'}`")

tab_url, tab_file = st.tabs(["🔗 URL (YouTube etc.)", "📁 Arquivo local"])

source_audio_path: Path | None = None
source_title: str | None = None
tmpdir_holder: list[str] = []

with tab_url:
    url = st.text_input("Cole a URL do vídeo", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("Baixar áudio", type="primary", disabled=not url, key="btn_dl"):
        if not is_url(url):
            st.error("URL inválida.")
        else:
            with st.spinner("Baixando áudio com yt-dlp..."):
                tmpdir = tempfile.mkdtemp(prefix="ytdlp_")
                tmpdir_holder.append(tmpdir)
                try:
                    audio_path, title = download_audio(url, Path(tmpdir))
                    st.session_state["audio_path"] = str(audio_path)
                    st.session_state["title"] = title
                    st.success(f"Áudio baixado: **{title}**")
                except Exception as e:
                    st.error(f"Falha ao baixar: {e}")

with tab_file:
    up = st.file_uploader(
        "Selecione um arquivo de áudio/vídeo",
        type=["mp3", "wav", "m4a", "ogg", "opus", "flac", "mp4", "mkv", "webm", "mov"],
    )
    if up is not None:
        tmpdir = tempfile.mkdtemp(prefix="upload_")
        out = Path(tmpdir) / up.name
        out.write_bytes(up.read())
        st.session_state["audio_path"] = str(out)
        st.session_state["title"] = Path(up.name).stem
        st.info(f"Arquivo pronto: **{up.name}**")

# Ação principal: transcrever
if "audio_path" in st.session_state:
    st.divider()
    st.subheader(f"📝 Transcrever: {st.session_state.get('title', '')}")
    audio_path = Path(st.session_state["audio_path"])
    if audio_path.exists():
        st.audio(str(audio_path))
    if st.button("▶️ Iniciar transcrição", type="primary"):
        model = load_model(model_size, device, compute)
        segs, info = run_transcription(audio_path, model)
        full_text = " ".join(s["text"] for s in segs).strip()

        st.success(f"✅ Concluído — idioma detectado: `{info.language}` ({info.language_probability:.0%})")

        dialogue_text = None
        speaker_segments = None
        if enable_diar and DIARIZATION_AVAILABLE:
            with st.spinner("🗣️ Identificando falantes com NeMo (primeira execução baixa ~600MB)..."):
                try:
                    speaker_segments = diarize_and_merge(audio_path, segs, num_speakers=num_speakers)
                    dialogue_text = format_as_dialogue(speaker_segments)
                    n_spk = len({s.speaker for s in speaker_segments})
                    st.success(f"✅ Diarization concluída — {n_spk} falante(s) identificado(s)")
                except Exception as e:
                    st.error(f"Falha na diarization: {e}")

        col1, col2 = st.columns([2, 1])
        with col1:
            if dialogue_text:
                st.markdown("### Diálogo (com falantes)")
                st.text_area("Diálogo", dialogue_text, height=400, label_visibility="collapsed")
                with st.expander("Ver texto puro (sem falantes)"):
                    st.text_area("Texto", full_text, height=200, label_visibility="collapsed", key="raw_text")
            else:
                st.markdown("### Texto")
                st.text_area("Transcrição", full_text, height=400, label_visibility="collapsed")
        with col2:
            st.markdown("### Downloads")
            base = safe_filename(st.session_state.get("title", "transcricao"))
            st.download_button(
                "⬇️ Baixar .txt",
                full_text + "\n",
                file_name=f"{base}.txt",
                mime="text/plain",
            )
            if dialogue_text:
                st.download_button(
                    "⬇️ Baixar diálogo (.txt)",
                    dialogue_text + "\n",
                    file_name=f"{base}_dialogo.txt",
                    mime="text/plain",
                )
            import json as _json
            payload = {"text": full_text, "segments": segs, "language": info.language}
            if speaker_segments:
                payload["speakers"] = [
                    {"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text}
                    for s in speaker_segments
                ]
            st.download_button(
                "⬇️ Baixar .json (com timestamps)",
                _json.dumps(payload, ensure_ascii=False, indent=2),
                file_name=f"{base}.json",
                mime="application/json",
            )

            st.markdown("### Segmentos")
            with st.expander(f"Ver {len(segs)} segmentos"):
                for s in segs:
                    st.markdown(f"`[{s['start']:.2f}s → {s['end']:.2f}s]` {s['text']}")
else:
    st.info("⬆️ Cole uma URL ou faça upload de um arquivo para começar.")
