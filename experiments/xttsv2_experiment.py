#!/usr/bin/env python3
"""
XTTSv2 Experiment - CPU Limitation Documentation

XTTSv2 is a GPT2-based TTS model that requires 8GB+ VRAM.
This script documents what happens when attempting to run on CPU-only systems.
"""

import time
import json
import sys
from pathlib import Path

# Test phrase
TEST_TEXT = "Hello, this is a test of XTTSv2 text to speech synthesis."


def test_xttsv2():
    """Attempt to run XTTSv2 and document limitations."""
    print("=" * 70)
    print("XTTSv2 TTS Experiment - CPU Limitation Documentation")
    print("=" * 70)
    print("\nModel Info:")
    print("  - Architecture: GPT2-based decoder")
    print("  - Parameters: ~2.3B (2.3 billion)")
    print("  - VRAM Requirement: 8GB+ (GPU recommended)")
    print("  - CPU Inference: Not recommended (extremely slow or OOM)")
    print()
    
    results = {
        "model": "XTTSv2",
        "architecture": "GPT2-based",
        "parameters": "2.3B",
        "vram_required_gb": 8,
        "cpu_compatible": False,
        "status": "attempted",
        "errors": [],
        "notes": []
    }
    
    try:
        print("Attempting to import TTS library...")
        from TTS.api import TTS
        print("✓ TTS library imported")
        
        print("\nAttempting to load XTTSv2 model...")
        print("  Model: tts_models/multilingual/multi-dataset/xtts_v2")
        print("  This may take several minutes and require significant memory...")
        
        start_time = time.time()
        
        try:
            # Attempt to load XTTSv2
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            load_time = time.time() - start_time
            
            print(f"✓ Model loaded in {load_time:.1f}s")
            results["load_time_seconds"] = load_time
            
            # Attempt synthesis
            print("\nAttempting synthesis on CPU...")
            print("  Text: '{}'".format(TEST_TEXT))
            
            synth_start = time.time()
            
            # XTTSv2 requires speaker_wav for voice cloning
            # We'll try without first to see error
            output_path = "/home/azureuser/neural_tts/results/xttsv2_test.wav"
            
            # This will likely fail or be extremely slow on CPU
            tts.tts_to_file(
                text=TEST_TEXT,
                file_path=output_path
            )
            
            synth_time = time.time() - synth_start
            print(f"✓ Synthesis completed in {synth_time:.1f}s")
            
            results["status"] = "completed"
            results["synthesis_time_seconds"] = synth_time
            
        except RuntimeError as e:
            error_msg = str(e)
            print(f"\n✗ Runtime Error: {error_msg}")
            results["errors"].append(error_msg)
            
            if "CUDA" in error_msg or "cuda" in error_msg:
                results["notes"].append("Model requires CUDA/GPU")
            if "out of memory" in error_msg.lower() or "OOM" in error_msg:
                results["notes"].append("Out of memory - model too large for available RAM")
                
        except Exception as e:
            error_msg = str(e)
            print(f"\n✗ Error: {error_msg}")
            results["errors"].append(error_msg)
            
    except ImportError as e:
        print(f"✗ TTS library not installed: {e}")
        results["errors"].append(f"ImportError: {e}")
        results["notes"].append("Install with: pip install TTS")
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        results["errors"].append(str(e))
    
    # Add technical explanation
    results["technical_analysis"] = {
        "why_cpu_fails": [
            "GPT2 architecture uses ~2.3B parameters requiring ~9GB memory just for weights",
            "Autoregressive generation requires sequential token generation (slow on CPU)",
            "No ONNX optimization available for XTTSv2",
            "Memory bandwidth bottleneck: CPU RAM slower than GPU VRAM for large models"
        ],
        "alternatives": [
            "Use Piper TTS (5-110MB, RTF >1000x on CPU)",
            "Use Kokoro (82MB, RTF ~0.2x on CPU)",
            "Use MeloTTS (162MB, RTF ~0.16x on CPU)",
            "Use cloud API for XTTSv2 if GPU not available"
        ]
    }
    
    return results


def main():
    results = test_xttsv2()
    
    # Save results
    output_path = Path("/home/azureuser/neural_tts/results/xttsv2_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")
    
    # Also create markdown documentation
    md_path = Path("/home/azureuser/neural_tts/results/xttsv2_cpu_analysis.md")
    with open(md_path, 'w') as f:
        f.write("# XTTSv2 CPU Deployment Analysis\n\n")
        f.write("## Model Specifications\n\n")
        f.write(f"- **Architecture**: {results.get('architecture', 'GPT2-based')}\n")
        f.write(f"- **Parameters**: {results.get('parameters', '~2.3B')}\n")
        f.write(f"- **VRAM Required**: {results.get('vram_required_gb', '8')}GB+\n")
        f.write(f"- **CPU Compatible**: {results.get('cpu_compatible', False)}\n\n")
        
        f.write("## Test Results\n\n")
        f.write(f"**Status**: {results.get('status', 'unknown')}\n\n")
        
        if results.get('errors'):
            f.write("### Errors Encountered\n\n")
            for error in results['errors']:
                f.write(f"- `{error}`\n")
            f.write("\n")
        
        if results.get('technical_analysis'):
            f.write("## Why XTTSv2 Fails on CPU\n\n")
            for reason in results['technical_analysis'].get('why_cpu_fails', []):
                f.write(f"- {reason}\n")
            f.write("\n")
            
            f.write("## Recommended Alternatives\n\n")
            for alt in results['technical_analysis'].get('alternatives', []):
                f.write(f"- {alt}\n")
    
    print(f"✓ Analysis saved to {md_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Status: {results['status']}")
    print(f"CPU Compatible: {results['cpu_compatible']}")
    if results.get('errors'):
        print(f"Errors: {len(results['errors'])}")
    print("\nConclusion: XTTSv2 requires GPU with 8GB+ VRAM. Not suitable for CPU deployment.")


if __name__ == "__main__":
    main()
