#!/usr/bin/env /usr/bin/python3
"""
Create comparison visualizations for Piper TTS vs MeloTTS
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
    
    # Load MeloTTS results
    with open(results_dir / "melotts_results.json") as f:
        melotts_results = json.load(f)
    
    # Prepare Piper data
    piper_qualities = []
    piper_sizes = []
    piper_rtfs = []
    
    for voice_name, data in piper_results.items():
        piper_qualities.append(data["quality"])
        piper_sizes.append(data["size_mb"])
        piper_rtfs.append(data["average_rtf"])
    
    # MeloTTS data
    melotts_size = melotts_results["model_size_mb"]
    melotts_rtf = melotts_results["average_rtf"]
    
    # Create comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Piper TTS vs MeloTTS: Performance Comparison on CPU', fontsize=14, fontweight='bold')
    
    # 1. Model Size Comparison
    ax1 = axes[0]
    
    # Piper bars
    colors_piper = ['#2ecc71', '#3498db', '#e74c3c']
    bars1 = ax1.bar([f'Piper\\n{q}' for q in piper_qualities], piper_sizes, 
                     color=colors_piper, edgecolor='black', linewidth=1.5, label='Piper TTS')
    
    # MeloTTS bar
    bars2 = ax1.bar(['MeloTTS\\nZH_MIX_EN'], [melotts_size], 
                     color='#9b59b6', edgecolor='black', linewidth=1.5, label='MeloTTS')
    
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('Model Size (MB)', fontsize=11)
    ax1.set_title('Model Size Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    
    # Add value labels
    for bar, size in zip(bars1, piper_sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.text(bars2[0].get_x() + bars2[0].get_width()/2., melotts_size,
            f'{melotts_size:.1f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. RTF Comparison (log scale)
    ax2 = axes[1]
    
    # Piper bars
    bars3 = ax2.bar([f'Piper\\n{q}' for q in piper_qualities], piper_rtfs, 
                     color=colors_piper, edgecolor='black', linewidth=1.5)
    
    # MeloTTS bar (note: MeloTTS RTF is much lower, so it will appear very small)
    bars4 = ax2.bar(['MeloTTS\\nZH_MIX_EN'], [melotts_rtf], 
                     color='#9b59b6', edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Model', fontsize=11)
    ax2.set_ylabel('Real-Time Factor (RTF) - Lower is Better', fontsize=11)
    ax2.set_title('Synthesis Speed Comparison (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, rtf in zip(bars3, piper_rtfs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rtf:.0f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.text(bars4[0].get_x() + bars4[0].get_width()/2., melotts_rtf,
            f'{melotts_rtf:.2f}x',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add explanation
    ax2.text(0.98, 0.98, 
            "RTF < 1: Faster than real-time\\nRTF > 1: Slower than real-time",
            transform=ax2.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(results_dir / 'tts_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Comparison visualization saved to {results_dir / 'tts_comparison.png'}")
    
    # Create summary table visualization
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    # Prepare table data
    table_data = [
        ['Model', 'Size (MB)', 'Avg RTF', 'Sample Rate', 'Status'],
        ['Piper Low', f'{piper_sizes[0]:.1f}', f'{piper_rtfs[0]:.0f}x', '22050 Hz', 'Faster than real-time'],
        ['Piper Medium', f'{piper_sizes[1]:.1f}', f'{piper_rtfs[1]:.0f}x', '22050 Hz', 'Faster than real-time'],
        ['Piper High', f'{piper_sizes[2]:.1f}', f'{piper_rtfs[2]:.0f}x', '22050 Hz', 'Faster than real-time'],
        ['MeloTTS ZH_MIX_EN', f'{melotts_size:.1f}', f'{melotts_rtf:.2f}x', '44100 Hz', 'Faster than real-time'],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    for i, color in enumerate(colors):
        table[(i+1, 0)].set_facecolor(color)
        table[(i+1, 0)].set_text_props(weight='bold', color='white')
    
    ax.set_title('TTS Model Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'comparison_table.png', dpi=150, bbox_inches='tight')
    print(f"Comparison table saved to {results_dir / 'comparison_table.png'}")
    
    print("\\nAll visualizations created successfully!")
    print(f"\\nKey findings:")
    print(f"  - Piper Low: {piper_sizes[0]:.1f}MB, RTF {piper_rtfs[0]:.0f}x")
    print(f"  - Piper Medium: {piper_sizes[1]:.1f}MB, RTF {piper_rtfs[1]:.0f}x")
    print(f"  - Piper High: {piper_sizes[2]:.1f}MB, RTF {piper_rtfs[2]:.0f}x")
    print(f"  - MeloTTS: {melotts_size:.1f}MB, RTF {melotts_rtf:.2f}x")

if __name__ == "__main__":
    main()
