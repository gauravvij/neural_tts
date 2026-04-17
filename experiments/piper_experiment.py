#!/usr/bin/env python3
"""
Piper TTS Experiment Script
Measures RTF (Real-Time Factor) for different quality tiers on CPU
"""

import os
import time
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

# Piper voice models to test
# Using lessac voice which has low, medium, and high qualities
PIPER_VOICES = {
    "en_US-lessac-low": {
        "voice_name": "lessac",
        "quality": "low",
        "size_mb": 5.8
    },
    "en_US-lessac-medium": {
        "voice_name": "lessac",
        "quality": "medium",
        "size_mb": 62.0
    },
    "en_US-lessac-high": {
        "voice_name": "lessac",
        "quality": "high",
        "size_mb": 110.0
    }
}

TEST_TEXTS = [
    "Hello, this is a test of neural text to speech synthesis.",
    "The quick brown fox jumps over the lazy dog.",
    "Neural networks have revolutionized speech synthesis.",
    "Real-time factor measures synthesis speed relative to audio duration.",
    "Modern TTS models can run efficiently on CPU."
]

def download_model(voice_name, voice_info, models_dir):
    """Download Piper voice model and config using huggingface_hub"""
    model_path = models_dir / f"{voice_name}.onnx"
    config_path = models_dir / f"{voice_name}.onnx.json"
    
    repo_id = "rhasspy/piper-voices"
    
    voice = voice_info["voice_name"]
    quality = voice_info["quality"]
    subfolder = f"en/en_US/{voice}/{quality}"
    model_filename = f"en_US-{voice}-{quality}.onnx"
    config_filename = f"en_US-{voice}-{quality}.onnx.json"
    
    if not model_path.exists():
        print(f"Downloading {voice_name} model...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
            subfolder=subfolder,
            revision="v1.0.0",
            local_dir=models_dir
        )
        # Rename to our naming convention
        os.rename(downloaded, model_path)
        print(f"  Saved to {model_path}")
    
    if not config_path.exists():
        print(f"Downloading {voice_name} config...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=config_filename,
            subfolder=subfolder,
            revision="v1.0.0",
            local_dir=models_dir
        )
        os.rename(downloaded, config_path)
        print(f"  Saved to {config_path}")
    
    return model_path, config_path

def measure_rtf(piper_tts, text, num_runs=3):
    """Measure Real-Time Factor for a given text"""
    rtf_values = []
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        # Piper returns a generator of audio chunks, convert to list
        audio_chunks = list(piper_tts.synthesize(text))
        end_time = time.perf_counter()
        
        # Concatenate all audio chunks
        if audio_chunks:
            # Handle zero-dimensional arrays by flattening first
            flattened = [np.atleast_1d(chunk) for chunk in audio_chunks]
            audio = np.concatenate(flattened)
        else:
            audio = np.array([])
        
        synthesis_time = end_time - start_time
        audio_duration = len(audio) / 22050  # Piper uses 22050 Hz
        rtf = synthesis_time / audio_duration if audio_duration > 0 else 0
        rtf_values.append(rtf)
    
    return {
        "rtf_mean": np.mean(rtf_values),
        "rtf_std": np.std(rtf_values),
        "rtf_values": rtf_values,
        "audio_duration": audio_duration,
        "synthesis_time": synthesis_time
    }

def main():
    from piper import PiperVoice
    
    models_dir = Path("/home/azureuser/neural_tts/models")
    results_dir = Path("/home/azureuser/neural_tts/results")
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    results = {}
    
    for voice_name, voice_info in PIPER_VOICES.items():
        print(f"\n{'='*60}")
        print(f"Testing {voice_name} ({voice_info['quality']} quality)")
        print(f"{'='*60}")
        
        # Download model
        model_path, config_path = download_model(voice_name, voice_info, models_dir)
        
        # Load voice
        print(f"Loading model...")
        voice = PiperVoice.load(str(model_path))
        
        # Test each text
        voice_results = {
            "quality": voice_info["quality"],
            "size_mb": voice_info["size_mb"],
            "tests": []
        }
        
        for i, text in enumerate(TEST_TEXTS):
            print(f"\nTest {i+1}/{len(TEST_TEXTS)}: '{text[:50]}...'")
            
            # Measure RTF
            rtf_data = measure_rtf(voice, text)
            
            print(f"  RTF: {rtf_data['rtf_mean']:.3f} ± {rtf_data['rtf_std']:.3f}")
            print(f"  Audio duration: {rtf_data['audio_duration']:.2f}s")
            print(f"  Synthesis time: {rtf_data['synthesis_time']:.3f}s")
            
            voice_results["tests"].append({
                "text": text,
                "rtf_mean": rtf_data["rtf_mean"],
                "rtf_std": rtf_data["rtf_std"],
                "audio_duration": rtf_data["audio_duration"],
                "synthesis_time": rtf_data["synthesis_time"]
            })
        
        # Calculate average RTF across all tests
        avg_rtf = np.mean([t["rtf_mean"] for t in voice_results["tests"]])
        voice_results["average_rtf"] = float(avg_rtf)
        
        results[voice_name] = voice_results
        
        print(f"\nAverage RTF for {voice_name}: {avg_rtf:.3f}")
    
    # Save results
    results_file = results_dir / "piper_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Piper TTS Experiment Complete!")
    print(f"Results saved to {results_file}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary:")
    for voice_name, data in results.items():
        print(f"  {voice_name}: RTF = {data['average_rtf']:.3f}, Size = {data['size_mb']} MB")

if __name__ == "__main__":
    main()
