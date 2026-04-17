# Neural TTS: From WaveNet to Whisper-Small — A Technical Journey

## Goal
Create a comprehensive, technically accurate blog post covering the evolution of Neural TTS from early models to modern lightweight CPU-friendly versions, with hands-on experimentation and Neo integration examples.

## Research Summary

### Neural TTS Evolution Timeline
- **2016**: WaveNet (DeepMind) - First neural raw waveform model, dilated causal convolutions, autoregressive, high quality but slow
- **2017**: Tacotron (Google) - End-to-end seq2seq with attention, mel-spectrogram prediction
- **2018**: Tacotron 2 - Improved with WaveNet vocoder, near-human quality
- **2019**: FastSpeech (Microsoft) - Non-autoregressive, parallel generation, 270x faster
- **2020**: FastSpeech 2 - Variance adaptor for pitch/energy/duration, no teacher-student needed
- **2021**: VITS - End-to-end with normalizing flows, adversarial training, parallel generation
- **2023-2025**: Modern lightweight models (MeloTTS, StyleTTS2, XTTS, Piper, Kokoro)

### Key Modern Models on HuggingFace
1. **MeloTTS** (MyShell.ai): VITS-based, real-time CPU inference, multilingual, mixed Chinese/English
2. **Kokoro**: 82M parameters, StyleTTS2-based, high quality on modest hardware
3. **Piper TTS**: ONNX-based, 5-32M parameters depending on quality tier, RTF < 1.0 on Raspberry Pi
4. **StyleTTS2**: ~300MB checkpoint, human-level naturalness, zero-shot voice cloning
5. **XTTSv2** (Coqui): GPT2-based decoder, 8GB+ VRAM recommended, 17+ languages
6. **Parler-TTS Mini**: 880M parameters, style control, SDPA/Flash Attention 2 support

### Model Size Evolution
- WaveNet: Large, autoregressive, GPU required
- Tacotron 2: Medium, GPU recommended
- FastSpeech 2: Medium, faster inference
- VITS: Compact, efficient
- Piper (x_low): 5-7M params, 16kHz
- Piper (low): 15-20M params, 16kHz
- Piper (medium): 15-20M params, 22.05kHz (~60MB)
- Piper (high): 28-32M params, 22.05kHz
- Kokoro: 82M params
- StyleTTS2: ~300MB

### Quality Factors
- **Prosody**: Rhythm, stress, intonation patterns
- **Pitch (F0)**: Fundamental frequency contours
- **Energy**: Loudness variations
- **Duration**: Timing of phonemes and pauses
- **MOS (Mean Opinion Score)**: 1-5 scale, human evaluation
- **RTF (Real-Time Factor)**: Processing time / audio duration (< 1.0 is real-time)

## Approach
Write a human-feeling technical blog that traces the actual evolution of Neural TTS with accurate technical details. Include hands-on experimentation comparing model sizes and RTF on CPU. Create visualizations showing the evolution. End with Neo integration examples.

## Subtasks

1. **Set up environment and install TTS libraries**
   - Install transformers, piper-tts, melotts, soundfile, scipy, matplotlib
   - Verify installations work
   - Expected output: Working Python environment with TTS libraries

2. **Run Piper TTS experiments with different quality tiers**
   - Download x_low, low, medium, high quality Piper models
   - Measure RTF on CPU for each tier
   - Record audio samples
   - Expected output: RTF measurements, audio files, comparison data

3. **Run MeloTTS experiment**
   - Install and run MeloTTS on CPU
   - Measure RTF
   - Record sample
   - Expected output: RTF measurement, audio file

4. **Create comparison visualizations**
   - Model size vs year chart
   - RTF comparison chart
   - Quality tier comparison
   - Expected output: PNG charts saved to /home/azureuser/neural_tts/

5. **Write the technical blog**
   - Section 1: What is Neural TTS (technical foundation)
   - Section 2: Evolution timeline (WaveNet to modern)
   - Section 3: Model size evolution and CPU efficiency
   - Section 4: What impacts quality (prosody, MOS, etc.)
   - Section 5: Hands-on experimentation results
   - Section 6: Building pipelines with Neo
   - Expected output: neural_tts_blog.md with all sections

6. **Create Neo integration examples**
   - Python script showing Neo building TTS pipeline
   - Example workflow code
   - Expected output: neo_tts_example.py

## Deliverables

| File Path | Description |
|-----------|-------------|
| /home/azureuser/neural_tts/neural_tts_blog.md | Complete technical blog |
| /home/azureuser/neural_tts/neo_tts_example.py | Neo integration examples |
| /home/azureuser/neural_tts/model_size_evolution.png | Chart showing parameter reduction over time |
| /home/azureuser/neural_tts/rtf_comparison.png | Real-time factor comparison chart |
| /home/azureuser/neural_tts/piper_samples/ | Audio samples from experiments |

## Evaluation Criteria
- Blog must be technically accurate with no hallucinated facts
- Human-feeling writing style (no em dashes, not salesy, not boring)
- Actual experimentation data included
- Visualizations based on real data
- Neo integration section with working code examples

## Notes
- No GPU available, focus on CPU-friendly models
- Piper TTS is ideal for CPU experimentation
- MeloTTS also optimized for CPU
- Blog should feel like written by a practitioner who actually ran the experiments
