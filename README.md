# Neural TTS: From WaveNet to Edge Deployment

A hands-on exploration of how neural text-to-speech models have evolved from resource-heavy autoregressive architectures to lightweight ONNX models that run in real-time on CPU. This repository contains reproducible experiments measuring synthesis speed, quality trade-offs, and practical deployment considerations.

## What This Repository Contains

- **Five TTS model experiments** with RTF (Real-Time Factor) measurements on CPU
- **Performance visualizations** comparing model size vs. synthesis speed
- **A technical deep-dive** covering the evolution from WaveNet (2016) to modern efficient architectures
- **Reproducible code** for running your own benchmarks

## Quick Start

```bash
# Clone the repository
git clone https://github.com/gauravvij/neural_tts.git
cd neural_tts

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run a quick test (Piper TTS low quality, ~6MB model)
python experiments/piper_experiment.py
```

## Understanding RTF (Real-Time Factor)

RTF measures how fast a model generates speech relative to the duration of that speech:

- **RTF = 1.0**: Generates 1 second of audio in 1 second (real-time)
- **RTF = 0.1**: Generates 1 second of audio in 0.1 seconds (10x faster than real-time)
- **RTF = 10.0**: Generates 1 second of audio in 10 seconds (too slow for real-time use)

For interactive applications, you typically want RTF < 0.5. For batch processing, anything under 2.0 is workable.

## Experiments Overview

### 1. Piper TTS (Recommended First Try)

Piper is the standout performer for CPU deployment. Three quality tiers let you choose your size/speed trade-off:

| Quality | Size | Average RTF | Use Case |
|---------|------|-------------|----------|
| Low | 5.8 MB | 1409x | Embedded devices, IoT |
| Medium | 62 MB | 2483x | General purpose |
| High | 110 MB | 7603x | High-quality offline |

```bash
python experiments/piper_experiment.py
```

Models download automatically from HuggingFace (`rhasspy/piper-voices`).

### 2. Kokoro TTS

A StyleTTS2-based model (82MB) that balances quality and speed. Good for applications where you want natural prosody without the bulk of larger models.

```bash
pip install kokoro soundfile torch
python experiments/kokoro_experiment.py
```

**Note**: Kokoro requires PyTorch. The first run downloads the model (~82MB).

### 3. MeloTTS

MeloTTS offers multilingual support with a clean ONNX export. The experiment uses the AXERA-TECH community models.

```bash
pip install melotts-onnx
python experiments/melotts_experiment.py
```

**Note**: The official MeloTTS-ONNX models are gated on HuggingFace. This experiment uses publicly available community models.

### 4. Parler-TTS Mini

A T5-based architecture (880MB) that generates high-quality speech but requires more compute. Useful for understanding the quality/speed trade-off of larger transformer models.

```bash
pip install parler-tts soundfile
python experiments/parler_experiment.py
```

**Note**: This model runs slower than real-time on CPU (RTF ~7x), making it better suited for GPU deployment or offline batch processing.

### 5. XTTSv2

A 2.3 billion parameter model that produces excellent quality but requires GPU (8GB+ VRAM). The experiment script documents the architecture and requirements rather than running on CPU.

```bash
python experiments/xttsv2_experiment.py
```

## Project Structure

```
neural_tts/
├── blog/
│   └── neural_tts_evolution.md    # Technical deep-dive
├── experiments/
│   ├── piper_experiment.py          # Fastest, best for CPU
│   ├── kokoro_experiment.py         # Balanced quality/speed
│   ├── melotts_experiment.py        # Multilingual
│   ├── parler_experiment.py         # High quality, slow
│   └── xttsv2_experiment.py         # GPU-only documentation
├── results/
│   ├── piper_results.json           # Raw experiment data
│   ├── kokoro_results.json
│   ├── *.png                        # Visualizations
│   └── experiment_summary.md        # Quick reference
├── models/                          # Downloaded models (gitignored)
└── plans/
    └── plan.md                      # Original research plan
```

## Requirements

### Hardware

- **CPU**: Any modern x86_64 processor (tested on 8-core)
- **RAM**: 4GB minimum, 8GB recommended for Parler-TTS
- **Storage**: ~2GB for all models
- **GPU**: Optional (only XTTSv2 requires it)

### Software

- Python 3.8+
- See `requirements.txt` for package dependencies

## Installation Details

### Core Dependencies

```bash
pip install piper-tts huggingface-hub soundfile numpy
```

### Optional Dependencies (per experiment)

```bash
# Kokoro
pip install kokoro torch

# MeloTTS
pip install melotts-onnx

# Parler-TTS
pip install parler-tts

# Visualizations
pip install matplotlib seaborn pandas
```

## Interpreting Results

Each experiment outputs a JSON file with:

- `rtf`: Real-Time Factor for each test phrase
- `audio_duration`: Length of generated audio in seconds
- `synthesis_time`: Time taken to generate in seconds

Example from `results/piper_results.json`:

```json
{
  "en_US-lessac-low": {
    "quality": "low",
    "size_mb": 5.8,
    "average_rtf": 1409.358,
    "tests": [...]
  }
}
```

An RTF of 1409 means the model generates speech 1409 times faster than real-time, or 1 second of audio in 0.7 milliseconds.

## Visualizations

Generate comparison charts after running experiments:

```bash
python experiments/create_comparison_visualizations.py
```

This produces:
- `rtf_comparison.png`: RTF across all models
- `model_evolution.png`: Historical size progression
- `piper_comparison.png`: Piper quality tier analysis

## Common Issues

### Model Download Failures

If HuggingFace downloads fail, check your internet connection and ensure `huggingface-hub` is up to date:

```bash
pip install -U huggingface-hub
```

### ONNX Runtime Errors

Some experiments require ONNX Runtime. Install the appropriate version for your system:

```bash
# CPU-only
pip install onnxruntime

# With GPU support (if available)
pip install onnxruntime-gpu
```

### Memory Errors with Parler-TTS

Parler-TTS requires significant RAM. If you encounter OOM errors:

1. Close other applications
2. Reduce batch size in the experiment script
3. Consider running on a machine with 16GB+ RAM

## Extending the Experiments

The experiment scripts are modular. To test your own text:

1. Edit the `TEST_PHRASES` list in any experiment file
2. Run the script
3. Check `results/` for updated JSON output

To add a new TTS model:

1. Create `experiments/your_model_experiment.py`
2. Follow the pattern: import, measure RTF, save JSON
3. Add visualization support in `create_comparison_visualizations.py`

## What We Learned

1. **Piper TTS is the practical choice** for CPU deployment. The low-quality tier (5.8MB) runs 1400x faster than real-time with surprisingly good output.

2. **Model size does not equal quality**. The 110MB Piper high-quality tier is not proportionally better than the 62MB medium tier, despite being 2x larger.

3. **ONNX export is transformative**. Models that started as PyTorch checkpoints (100MB+) become deployable artifacts (5-60MB) with no quality loss.

4. **Transformer TTS (Parler, XTTSv2) remains GPU-bound**. The quality is excellent, but CPU inference is 7-10x slower than real-time.

Read the full analysis in `blog/neural_tts_evolution.md`.

## Citation

If you use this work in research, please cite:

```bibtex
@misc{neural_tts_benchmark,
  title={Neural TTS: From WaveNet to Edge Deployment},
  author={Gaurav Vij},
  year={2026},
  howpublished={\url{https://github.com/gauravvij/neural_tts}}
}
```

## License

MIT License. See LICENSE file for details.

Model weights are subject to their respective licenses:
- Piper TTS: MIT License
- Kokoro: Apache 2.0
- MeloTTS: MIT License
- Parler-TTS: Apache 2.0
- XTTSv2: CPML License

## Contributing

Found an issue or want to add another TTS model? Open a pull request. The experiment structure is designed to be extensible.

---

**Note**: This repository contains no model weights (they exceed GitHub's file size limits). Models download automatically on first run via HuggingFace Hub.
