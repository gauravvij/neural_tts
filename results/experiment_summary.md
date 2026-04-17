# Neural TTS Experiment Summary

## Completed Experiments

### Piper TTS (Completed)
Successfully tested three quality tiers of Piper TTS on CPU:

| Quality | Model Size | Average RTF | Synthesis Time |
|---------|------------|-------------|----------------|
| Low | 5.8 MB | 1409x | ~8ms |
| Medium | 62 MB | 2483x | ~14ms |
| High | 110 MB | 7603x | ~43ms |

All models run significantly faster than real-time on CPU, making them suitable for edge deployment.

### MeloTTS (Blocked)
Attempted to run MeloTTS experiments but encountered access restrictions:

1. **myshell-ai/MeloTTS-ONNX**: Gated repository requiring authentication
2. **myshell-ai/MeloTTS-English-v3**: Contains checkpoint.pth but no ONNX export
3. **AXERA-TECH/MeloTTS**: Provides encoder/decoder split architecture with .axmodel format
   - Downloaded encoder-en.onnx, decoder-en.axmodel, g-en.bin
   - The .axmodel format is specific to AXERA hardware and not compatible with standard ONNX Runtime

The melotts-onnx package (v0.0.2) expects a single combined tts_model.onnx file with configuration.json,
which is not publicly available without authentication.

## Visualizations Created

1. **piper_comparison.png**: Four-panel comparison of Piper TTS quality tiers
   - Model size by quality tier
   - RTF by quality tier (log scale)
   - Synthesis time comparison
   - Size vs performance trade-off

2. **model_evolution.png**: Timeline of TTS model size evolution (2016-2022)
   - Shows progression from WaveNet (64MB) to Piper Low (5.8MB)
   - Color-coded by era: Autoregressive, Non-autoregressive, Efficient ONNX

3. **rtf_comparison.png**: RTF comparison bar chart for Piper TTS

## Key Findings

1. **Model Size vs Quality**: Piper's low-quality tier (5.8MB) provides surprisingly good speech quality
2. **CPU Performance**: All Piper models run 1000x+ faster than real-time on CPU
3. **Edge Deployment**: ONNX export enables efficient edge inference without GPU
4. **Accessibility**: Modern TTS is now viable on resource-constrained devices

## Files Generated

- `/home/azureuser/neural_tts/results/piper_results.json`: Raw experiment data
- `/home/azureuser/neural_tts/results/piper_comparison.png`: Performance comparison charts
- `/home/azureuser/neural_tts/results/model_evolution.png`: Historical evolution chart
- `/home/azureuser/neural_tts/results/rtf_comparison.png`: RTF comparison chart
- `/home/azureuser/neural_tts/blog/neural_tts_evolution.md`: Technical blog post
- `/home/azureuser/neural_tts/neo_integration/tts_pipeline.py`: Neo integration example
