#!/usr/bin/env /usr/bin/python3
"""
Create visualizations for TTS comparison
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    results_dir = Path("/home/azureuser/neural_tts/results")
    
    # Load Piper results
    with open(results_dir / "piper_results.json") as f:
        piper_results = json.load(f)
    
    # Prepare data
    qualities = []
    model_sizes = []
    avg_rtfs = []
    synthesis_times = []
    
    for voice_name, data in piper_results.items():
        qualities.append(data["quality"])
        model_sizes.append(data["size_mb"])
        avg_rtfs.append(data["average_rtf"])
        # Average synthesis time across tests
        avg_synth_time = np.mean([t["synthesis_time"] for t in data["tests"]])
        synthesis_times.append(avg_synth_time)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Neural TTS Performance Analysis: Piper TTS Quality Tiers', fontsize=14, fontweight='bold')
    
    # 1. Model Size vs Quality
    ax1 = axes[0, 0]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars1 = ax1.bar(qualities, model_sizes, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Quality Tier', fontsize=11)
    ax1.set_ylabel('Model Size (MB)', fontsize=11)
    ax1.set_title('Model Size by Quality Tier', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar, size in zip(bars1, model_sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.1f} MB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. RTF Comparison (log scale)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(qualities, avg_rtfs, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Quality Tier', fontsize=11)
    ax2.set_ylabel('Real-Time Factor (RTF)', fontsize=11)
    ax2.set_title('Synthesis Speed (RTF) by Quality Tier', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar, rtf in zip(bars2, avg_rtfs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rtf:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Synthesis Time Comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(qualities, synthesis_times, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Quality Tier', fontsize=11)
    ax3.set_ylabel('Average Synthesis Time (seconds)', fontsize=11)
    ax3.set_title('Synthesis Time by Quality Tier', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar, stime in zip(bars3, synthesis_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{stime*1000:.0f}ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Size vs Performance Trade-off
    ax4 = axes[1, 1]
    scatter = ax4.scatter(model_sizes, avg_rtfs, s=200, c=colors, 
                          edgecolors='black', linewidths=2, alpha=0.8)
    for i, quality in enumerate(qualities):
        ax4.annotate(quality, (model_sizes[i], avg_rtfs[i]), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold')
    ax4.set_xlabel('Model Size (MB)', fontsize=11)
    ax4.set_ylabel('Real-Time Factor (RTF)', fontsize=11)
    ax4.set_title('Size vs Performance Trade-off', fontsize=12, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'piper_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {results_dir / 'piper_comparison.png'}")
    
    # Create model evolution timeline
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    # Historical TTS model data (approximate)
    models = [
        ("WaveNet (2016)", 2016, 64, "Autoregressive WaveNet", "Very Slow"),
        ("Tacotron (2017)", 2017, 25, "Seq2seq + Griffin-Lim", "Slow"),
        ("Tacotron 2 (2018)", 2018, 30, "Seq2seq + WaveNet", "Slow"),
        ("FastSpeech (2019)", 2019, 50, "Non-autoregressive", "Fast"),
        ("FastSpeech 2 (2020)", 2020, 55, "Improved Fastspeech", "Fast"),
        ("Piper Low (2022)", 2022, 5.8, "VITS-based ONNX", "Very Fast"),
        ("Piper Medium (2022)", 2022, 62, "VITS-based ONNX", "Fast"),
        ("Piper High (2022)", 2022, 110, "VITS-based ONNX", "Medium"),
    ]
    
    years = [m[1] for m in models]
    sizes = [m[2] for m in models]
    names = [m[0] for m in models]
    
    # Color by era
    colors_timeline = []
    for year in years:
        if year <= 2017:
            colors_timeline.append('#e74c3c')  # Early era - red
        elif year <= 2019:
            colors_timeline.append('#f39c12')  # Middle era - orange
        else:
            colors_timeline.append('#2ecc71')  # Modern era - green
    
    ax.scatter(years, sizes, s=200, c=colors_timeline, edgecolors='black', linewidths=2, alpha=0.8, zorder=3)
    
    # Add connecting line
    ax.plot(years, sizes, 'k--', alpha=0.3, zorder=1)
    
    # Add labels
    for i, (name, year, size, arch, speed) in enumerate(models):
        offset = (15, 15) if i % 2 == 0 else (15, -25)
        ax.annotate(f"{name}\\n{size}MB", (year, size), 
                   xytext=offset, textcoords='offset points',
                   fontsize=9, ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Model Size (MB)', fontsize=12)
    ax.set_title('Neural TTS Model Size Evolution (2016-2022)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, zorder=0)
    ax.set_xlim(2015, 2024)
    ax.set_ylim(0, 130)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Autoregressive Era (2016-2017)'),
        Patch(facecolor='#f39c12', label='Non-Autoregressive Era (2018-2020)'),
        Patch(facecolor='#2ecc71', label='Efficient ONNX Era (2021+)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'model_evolution.png', dpi=150, bbox_inches='tight')
    print(f"Evolution chart saved to {results_dir / 'model_evolution.png'}")
    
    # Create RTF comparison bar chart
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    # RTF data (lower is better - means faster than real-time)
    # Note: These are the measured RTF values from our experiments
    rtf_data = [
        ("Piper Low", 1409),
        ("Piper Medium", 2483),
        ("Piper High", 7603),
    ]
    
    names = [d[0] for d in rtf_data]
    rtfs = [d[1] for d in rtf_data]
    
    bars = ax.barh(names, rtfs, color=['#2ecc71', '#3498db', '#e74c3c'], 
                   edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Real-Time Factor (RTF) - Lower is Better', fontsize=12)
    ax.set_title('CPU Synthesis Speed Comparison\\n(RTF = synthesis time / audio duration)', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, rtf in zip(bars, rtfs):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f' {rtf:.0f}x',
               ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Add explanation text
    ax.text(0.98, 0.02, 
           "RTF < 1: Faster than real-time\\nRTF > 1: Slower than real-time",
           transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(results_dir / 'rtf_comparison.png', dpi=150, bbox_inches='tight')
    print(f"RTF comparison saved to {results_dir / 'rtf_comparison.png'}")
    
    print("\\nAll visualizations created successfully!")

if __name__ == "__main__":
    main()
