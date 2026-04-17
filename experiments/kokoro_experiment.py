#!/usr/bin/env python3
"""
Kokoro TTS Experiment

Tests Kokoro TTS (82M parameters) on CPU.
Kokoro is a StyleTTS2-based model known for high quality on modest hardware.
"""

import time
import json
import numpy as np
from pathlib import Path
import sys

# Test phrases for consistent comparison
TEST_PHRASES = [
    "Hello, this is a test of neural text to speech synthesis.",
    "The quick brown fox jumps over the lazy dog.",
    "Neural networks have revolutionized speech synthesis.",
    "Real-time factor measures synthesis speed relative to audio duration.",
    "Modern TTS models can run efficiently on CPU."
]


def measure_kokoro():
    """Run Kokoro TTS experiments."""
    print("=" * 70)
    print("Kokoro TTS Experiment")
    print("=" * 70)
    
    try:
        from kokoro import KPipeline
        import soundfile as sf
        import torch
        print("✓ Kokoro imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Kokoro: {e}")
        return None
    
    results = {
        "model": "Kokoro-82M",
        "model_size_mb": 82.0,
        "sample_rate": 24000,
        "language": "en",
        "tests": []
    }
    
    try:
        # Initialize Kokoro pipeline
        # Kokoro uses a generator-based API that returns phonemes and audio
        print("\nInitializing Kokoro pipeline...")
        pipeline = KPipeline(lang_code='a')  # 'a' for American English
        print("✓ Pipeline initialized")
        
        # Run tests
        for i, text in enumerate(TEST_PHRASES, 1):
            print(f"\nTest {i}/{len(TEST_PHRASES)}: '{text[:50]}...'")
            
            try:
                # Measure synthesis time
                start_time = time.perf_counter()
                
                # Kokoro returns a generator of (graphemes, phonemes, audio) tuples
                # We need to consume the generator and collect audio
                generator = pipeline(text, voice='af_heart')  # American female voice
                
                # Collect all audio chunks from the generator
                audio_chunks = []
                for graphemes, phonemes, audio in generator:
                    if audio is not None and len(audio) > 0:
                        audio_chunks.append(audio)
                
                synthesis_time = time.perf_counter() - start_time
                
                # Concatenate audio chunks
                if audio_chunks:
                    audio_data = np.concatenate(audio_chunks)
                    sample_rate = 24000  # Kokoro default
                    
                    # Calculate audio duration
                    audio_duration = len(audio_data) / sample_rate
                    
                    # Calculate RTF
                    rtf = synthesis_time / audio_duration if audio_duration > 0 else 0
                    
                    print(f"  Audio duration: {audio_duration:.3f}s")
                    print(f"  Synthesis time: {synthesis_time:.4f}s")
                    print(f"  RTF: {rtf:.4f}x")
                    
                    # Save audio for first test
                    if i == 1:
                        output_path = Path("/home/azureuser/neural_tts/results/kokoro_sample.wav")
                        sf.write(output_path, audio_data, sample_rate)
                        print(f"  ✓ Sample saved to {output_path}")
                    
                    results["tests"].append({
                        "text": text,
                        "rtf": rtf,
                        "audio_duration": audio_duration,
                        "synthesis_time": synthesis_time
                    })
                else:
                    print(f"  ✗ No audio generated")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Calculate average RTF
        if results["tests"]:
            avg_rtf = sum(t["rtf"] for t in results["tests"]) / len(results["tests"])
            results["average_rtf"] = avg_rtf
            print(f"\n{'='*70}")
            print(f"Average RTF: {avg_rtf:.4f}x")
            print(f"{'='*70}")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Failed to run Kokoro experiment: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    results = measure_kokoro()
    
    if results:
        # Save results
        output_path = Path("/home/azureuser/neural_tts/results/kokoro_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
    else:
        print("\n✗ Experiment failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
