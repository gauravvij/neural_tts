#!/usr/bin/env /usr/bin/python3
"""
MeloTTS ONNX Experiment Script
Measures RTF (Real-Time Factor) on CPU using melotts-onnx package
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np

# Add melotts-onnx to path if needed
try:
    from melo_onnx import MeloTTS_ONNX
except ImportError:
    print("Error: melotts-onnx package not installed")
    print("Run: pip install melotts-onnx")
    sys.exit(1)

# Test texts - using Chinese-English mixed since the model supports both
TEST_TEXTS = [
    "Hello, this is a test of neural text to speech synthesis.",
    "The quick brown fox jumps over the lazy dog.",
    "Neural networks have revolutionized speech synthesis.",
    "Real-time factor measures synthesis speed relative to audio duration.",
    "Modern TTS models can run efficiently on CPU."
]

def measure_rtf(tts_model, text, speaker="EN", num_runs=3):
    """Measure Real-Time Factor for a given text"""
    rtf_values = []
    durations = []
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        # Run synthesis
        audio = tts_model.speak(text, speaker=speaker)
        end_time = time.perf_counter()
        
        synthesis_time = end_time - start_time
        
        # Calculate audio duration from sample rate
        sample_rate = tts_model.sample_rate
        audio_duration = len(audio) / sample_rate
        
        rtf = synthesis_time / audio_duration if audio_duration > 0 else 0
        rtf_values.append(rtf)
        durations.append(audio_duration)
    
    return {
        "rtf_mean": np.mean(rtf_values),
        "rtf_std": np.std(rtf_values),
        "rtf_values": rtf_values,
        "audio_duration": np.mean(durations),
        "synthesis_time": synthesis_time
    }

def main():
    model_path = "/home/azureuser/neural_tts/models/melotts_onnx"
    results_dir = Path("/home/azureuser/neural_tts/results")
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("MeloTTS ONNX Experiment")
    print("=" * 60)
    
    # Check if model files exist
    tts_model_file = Path(model_path) / "tts_model.onnx"
    config_file = Path(model_path) / "configuration.json"
    
    if not tts_model_file.exists():
        print(f"Error: Model file not found at {tts_model_file}")
        print("Please download the model first:")
        print("  huggingface-cli download seasonstudio/melotts_zh_mix_en_onnx --local-dir models/melotts_onnx")
        return
    
    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        return
    
    model_size_mb = tts_model_file.stat().st_size / (1024 * 1024)
    print(f"Model size: {model_size_mb:.1f} MB")
    print(f"Model path: {model_path}")
    
    # Load model
    print("\nLoading MeloTTS ONNX model...")
    try:
        tts = MeloTTS_ONNX(model_path, execution_provider="CPU", verbose=True)
        print(f"Model loaded successfully!")
        print(f"  Sample rate: {tts.sample_rate} Hz")
        print(f"  Available speakers: {tts.speakers}")
        print(f"  Language: {tts.language}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run tests
    results = {
        "model": "melotts_zh_mix_en_onnx",
        "model_size_mb": model_size_mb,
        "sample_rate": tts.sample_rate,
        "language": tts.language,
        "speakers": tts.speakers,
        "tests": []
    }
    
    # Use EN speaker for English text
    speaker = "EN" if "EN" in tts.speakers else tts.speakers[0]
    print(f"\nUsing speaker: {speaker}")
    
    print("\nRunning synthesis tests:")
    print("-" * 60)
    
    for i, text in enumerate(TEST_TEXTS):
        print(f"\nTest {i+1}/{len(TEST_TEXTS)}: '{text[:50]}...'")
        
        try:
            rtf_data = measure_rtf(tts, text, speaker=speaker)
            
            print(f"  RTF: {rtf_data['rtf_mean']:.3f} ± {rtf_data['rtf_std']:.3f}")
            print(f"  Audio duration: {rtf_data['audio_duration']:.2f}s")
            print(f"  Synthesis time: {rtf_data['synthesis_time']:.3f}s")
            
            results["tests"].append({
                "text": text,
                "rtf_mean": float(rtf_data["rtf_mean"]),
                "rtf_std": float(rtf_data["rtf_std"]),
                "audio_duration": float(rtf_data["audio_duration"]),
                "synthesis_time": float(rtf_data["synthesis_time"])
            })
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate average RTF
    if results["tests"]:
        avg_rtf = np.mean([t["rtf_mean"] for t in results["tests"]])
        results["average_rtf"] = float(avg_rtf)
        print(f"\nAverage RTF: {avg_rtf:.3f}")
    
    # Save results
    results_file = results_dir / "melotts_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
