#!/usr/bin/env python3
"""
Parler-TTS Mini Experiment

Tests Parler-TTS Mini (880M parameters) on CPU.
Parler-TTS uses SDPA/Flash Attention 2 for efficient inference.
"""

import time
import json
import numpy as np
from pathlib import Path
import sys
import torch

# Test phrases for consistent comparison
TEST_PHRASES = [
    "Hello, this is a test of neural text to speech synthesis.",
    "The quick brown fox jumps over the lazy dog.",
    "Neural networks have revolutionized speech synthesis.",
    "Real-time factor measures synthesis speed relative to audio duration.",
    "Modern TTS models can run efficiently on CPU."
]

# Style description for Parler-TTS
STYLE_DESCRIPTION = "Jon's voice is monotone yet fast in delivery, with a very close recording that captures every nuance of his speech."


def measure_parler():
    """Run Parler-TTS Mini experiments."""
    print("=" * 70)
    print("Parler-TTS Mini Experiment")
    print("=" * 70)
    
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        import soundfile as sf
        print("✓ Parler-TTS imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import required libraries: {e}")
        return None
    
    results = {
        "model": "Parler-TTS-Mini",
        "model_size_mb": 880.0,  # Approximate
        "sample_rate": 44100,
        "language": "en",
        "tests": []
    }
    
    try:
        # Load model and tokenizer
        print("\nLoading Parler-TTS Mini model...")
        print("  Model: parler-tts/parler-tts-mini-v1")
        print("  This may take a few minutes (model is ~880MB)...")
        
        model_name = "parler-tts/parler-tts-mini-v1"
        
        # Load model and tokenizer
        device = "cpu"
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("✓ Model loaded successfully")
        
        # Run tests
        for i, text in enumerate(TEST_PHRASES, 1):
            print(f"\nTest {i}/{len(TEST_PHRASES)}: '{text[:50]}...'")
            
            try:
                # Prepare inputs
                description_inputs = tokenizer(
                    [STYLE_DESCRIPTION],
                    return_tensors="pt"
                ).to(device)
                
                prompt_inputs = tokenizer(
                    [text],
                    return_tensors="pt"
                ).to(device)
                
                # Measure synthesis time
                start_time = time.perf_counter()
                
                # Generate audio
                with torch.no_grad():
                    generation = model.generate(
                        input_ids=description_inputs.input_ids,
                        attention_mask=description_inputs.attention_mask,
                        prompt_input_ids=prompt_inputs.input_ids,
                        prompt_attention_mask=prompt_inputs.attention_mask
                    )
                
                synthesis_time = time.perf_counter() - start_time
                
                # Extract audio
                audio_data = generation.cpu().numpy().squeeze()
                sample_rate = model.config.sampling_rate
                
                # Calculate audio duration
                audio_duration = len(audio_data) / sample_rate
                
                # Calculate RTF
                rtf = synthesis_time / audio_duration if audio_duration > 0 else 0
                
                print(f"  Audio duration: {audio_duration:.3f}s")
                print(f"  Synthesis time: {synthesis_time:.4f}s")
                print(f"  RTF: {rtf:.4f}x")
                
                # Save audio for first test
                if i == 1:
                    output_path = Path("/home/azureuser/neural_tts/results/parler_sample.wav")
                    sf.write(output_path, audio_data, sample_rate)
                    print(f"  ✓ Sample saved to {output_path}")
                
                results["tests"].append({
                    "text": text,
                    "rtf": rtf,
                    "audio_duration": audio_duration,
                    "synthesis_time": synthesis_time
                })
                
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
        print(f"\n✗ Failed to run Parler-TTS experiment: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    results = measure_parler()
    
    if results:
        # Save results
        output_path = Path("/home/azureuser/neural_tts/results/parler_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
    else:
        print("\n✗ Experiment failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
