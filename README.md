# soundapp

Transcrição local de áudio/vídeo (YouTube + upload) com **faster-whisper**, e diarização opcional via **NeMo**. 100% local, sem token, sem nuvem.

## Modelos

Os pesos (~2.1 GB) não ficam no git — estão em um [GitHub Release](https://github.com/athospugliese/soundapp/releases/tag/models-v1). Baixe e extraia em `models/` com:

```bash
bash download_models.sh
```

Sem isso, o app baixa os modelos automaticamente do Hugging Face / NGC na primeira execução (o código usa `models/` se existir, com fallback para download).

## Rodar

```bash
streamlit run app.py
```

## Estrutura

- `app.py` — UI Streamlit (transcrição)
- `transcribe_local.py` — download de áudio (yt-dlp) e helpers
- `diarize.py` — diarização via NeMo (identificação de falantes)
- `models/` — modelos locais (ignorado pelo git; ver acima)