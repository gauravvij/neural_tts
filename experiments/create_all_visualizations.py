#!/usr/bin/env /usr/bin/python3
"""Create comprehensive comparison visualizations for all TTS models"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    results_dir = Path("/home/azureuser/neural_tts/results")
    
    # Load all results
    with open(results_dir / "piper_results.json") as f:
        piper_results = json.load(f)
    with open(results_dir / "melotts_results.json") as f:
        melotts_results = json.load(f)
    with open(results_dir / "kokoro_results.json") as f:
        kokoro_results = json.load(f)
    with open(results_dir / "parler_results.json") as f:
        parler_results = json.load(f)
    
    # Prepare Piper data
    piper_sizes = []
    piper_rtfs = []
    for voice_name, data in piper_results.items():
        piper_sizes.append(data["size_mb"])
        piper_rtfs.append(data["average_rtf"])
    
    melotts_size = melotts_results["model_size_mb"]
    melotts_rtf = melotts_results["average_rtf"]
    kokoro_size = kokoro_results["model_size_mb"]
    kokoro_rtf = kokoro_results["average_rtf"]
    parler_size = parler_results["model_size_mb"]
    parler_rtf = parler_results["average_rtf"]
    
    # Create comprehensive comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Neural TTS Models: Complete Performance Comparison on CPU', fontsize=16, fontweight='bold')
    
    # 1. Model Size Comparison
    ax1 = axes[0, 0]
    models = ['Piper Low', 'Piper Med', 'Piper High', 'MeloTTS', 'Kokoro', 'Parler-TTS', 'XTTSv2*']
    sizes = [piper_sizes[0], piper_sizes[1], piper_sizes[2], melotts_size, kokoro_size, parler_size, 4500]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#95a5a6']
    
    bars1 = ax1.bar(models, sizes, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Model Size (MB)', fontsize=11)
    ax1.set_title('Model Size (Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. RTF Comparison
    ax2 = axes[0, 1]
    rtf_models = ['Piper Low', 'Piper Med', 'Piper High', 'MeloTTS', 'Kokoro', 'Parler-TTS']
    rtfs = [piper_rtfs[0], piper_rtfs[1], piper_rtfs[2], melotts_rtf, kokoro_rtf, parler_rtf]
    
    bars2 = ax2.bar(rtf_models, rtfs, color=colors[:6], edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Real-Time Factor (RTF)', fontsize=11)
    ax2.set_title('Synthesis Speed (Log Scale)\nLower is Better', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Real-time')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Size vs RTF Scatter
    ax3 = axes[1, 0]
    scatter_models = ['Piper Low', 'Piper Med', 'Piper High', 'MeloTTS', 'Kokoro', 'Parler-TTS']
    scatter_sizes = [piper_sizes[0], piper_sizes[1], piper_sizes[2], melotts_size, kokoro_size, parler_size]
    scatter_rtfs = [piper_rtfs[0], piper_rtfs[1], piper_rtfs[2], melotts_rtf, kokoro_rtf, parler_rtf]
    
    for i, (model, size, rtf, color) in enumerate(zip(scatter_models, scatter_sizes, scatter_rtfs, colors[:6])):
        ax3.scatter(size, rtf, s=200, c=color, edgecolors='black', linewidths=2, label=model, zorder=5)
        ax3.annotate(model, (size, rtf), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('Model Size (MB)', fontsize=11)
    ax3.set_ylabel('RTF', fontsize=11)
    ax3.set_title('Size vs Speed Trade-off', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 4. Comparison Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [
        ['Model', 'Size', 'RTF', 'Arch', 'CPU'],
        ['Piper Low', f'{piper_sizes[0]:.1f}MB', f'{piper_rtfs[0]:.0f}x', 'ONNX', 'Excellent'],
        ['Piper Med', f'{piper_sizes[1]:.1f}MB', f'{piper_rtfs[1]:.0f}x', 'ONNX', 'Excellent'],
        ['Piper High', f'{piper_sizes[2]:.1f}MB', f'{piper_rtfs[2]:.0f}x', 'ONNX', 'Excellent'],
        ['MeloTTS', f'{melotts_size:.1f}MB', f'{melotts_rtf:.2f}x', 'VITS', 'Excellent'],
        ['Kokoro', f'{kokoro_size:.1f}MB', f'{kokoro_rtf:.2f}x', 'StyleTTS2', 'Good'],
        ['Parler-TTS', f'{parler_size:.0f}MB', f'{parler_rtf:.1f}x', 'T5+DAC', 'Slow'],
        ['XTTSv2', '~4.5GB', 'N/A', 'GPT2', 'GPU Only'],
    ]
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.2, 0.15, 0.15, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Complete Model Comparison', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'tts_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: tts_comparison.png")
    
    # RTF bar chart
    fig2, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(rtf_models, rtfs, color=colors[:6], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Real-Time Factor (RTF)', fontsize=12)
    ax.set_title('TTS Model RTF Comparison on CPU (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Real-time threshold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    for bar, rtf in zip(bars, rtfs):
        height = bar.get_height()
        label = f'{rtf:.2f}x' if rtf < 1 else f'{rtf:.0f}x'
        ax.text(bar.get_x() + bar.get_width()/2., height, label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'rtf_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: rtf_comparison.png")
    
    print("\nAll visualizations created!")
    print(f"  MeloTTS: {melotts_rtf:.2f}x (fastest)")
    print(f"  Kokoro: {kokoro_rtf:.2f}x")
    print(f"  Parler-TTS: {parler_rtf:.1f}x (slowest)")

if __name__ == "__main__":
    main()
