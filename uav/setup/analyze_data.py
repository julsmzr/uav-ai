import os
import json
from collections import defaultdict

def analyze_data(statstic_json_filepath: str) -> None:
    with open(statstic_json_filepath, "r") as f:
        data = json.load(f)

    total_frames_vz, total_frames_ir = 0, 0
    total_amount_annotations_vz, total_amount_annotations_ir = 0, 0
    
    # Additional metrics
    sequence_stats = []
    annotation_density_vz = []
    annotation_density_ir = []
    frames_per_sequence = []
    annotations_per_sequence = []
    
    for sequence_name, (amount_frames_vz, amount_frames_ir, amount_annotations_vz, amount_annotations_ir) in data.items():
        total_frames_vz += amount_frames_vz
        total_frames_ir += amount_frames_ir
        total_amount_annotations_vz += amount_annotations_vz
        total_amount_annotations_ir += amount_annotations_ir
        
        # Calculate densities
        density_vz = amount_annotations_vz / amount_frames_vz if amount_frames_vz > 0 else 0
        density_ir = amount_annotations_ir / amount_frames_ir if amount_frames_ir > 0 else 0
        
        annotation_density_vz.append(density_vz)
        annotation_density_ir.append(density_ir)
        frames_per_sequence.append(amount_frames_vz + amount_frames_ir)
        annotations_per_sequence.append(amount_annotations_vz + amount_annotations_ir)
        
        sequence_stats.append({
            'name': sequence_name,
            'frames_vz': amount_frames_vz,
            'frames_ir': amount_frames_ir,
            'annotations_vz': amount_annotations_vz,
            'annotations_ir': amount_annotations_ir,
            'density_vz': density_vz,
            'density_ir': density_ir
        })

    # Basic totals
    print("=== BASIC STATISTICS ===")
    print("Total Frames VZ:", total_frames_vz)
    print("Total Frames IR:", total_frames_ir)
    print("Total Frames (All):", total_frames_vz + total_frames_ir)
    print()
    print("Total Amount Annotations VZ:", total_amount_annotations_vz)
    print("Total Amount Annotations IR:", total_amount_annotations_ir)
    print("Total Annotations (All):", total_amount_annotations_vz + total_amount_annotations_ir)
    print()

    # Annotation density analysis
    print("=== ANNOTATION DENSITY ===")
    print(f"Average annotations per frame VZ: {total_amount_annotations_vz/total_frames_vz:.3f}" if total_frames_vz > 0 else "No VZ frames")
    print(f"Average annotations per frame IR: {total_amount_annotations_ir/total_frames_ir:.3f}" if total_frames_ir > 0 else "No IR frames")
    print(f"Overall annotation density: {(total_amount_annotations_vz + total_amount_annotations_ir)/(total_frames_vz + total_frames_ir):.3f}")
    print()

    # Sequence statistics
    print("=== SEQUENCE ANALYSIS ===")
    print(f"Total sequences: {len(data)}")
    print(f"Average frames per sequence: {sum(frames_per_sequence)/len(frames_per_sequence):.1f}")
    print(f"Average annotations per sequence: {sum(annotations_per_sequence)/len(annotations_per_sequence):.1f}")
    print(f"Min frames in sequence: {min(frames_per_sequence)}")
    print(f"Max frames in sequence: {max(frames_per_sequence)}")
    print(f"Min annotations in sequence: {min(annotations_per_sequence)}")
    print(f"Max annotations in sequence: {max(annotations_per_sequence)}")
    print()

    # Find sequences with extreme characteristics
    most_dense_vz = max(sequence_stats, key=lambda x: x['density_vz'])
    most_dense_ir = max(sequence_stats, key=lambda x: x['density_ir'])
    least_dense_vz = min(sequence_stats, key=lambda x: x['density_vz'])
    least_dense_ir = min(sequence_stats, key=lambda x: x['density_ir'])
    
    print("=== EXTREME CASES ===")
    print(f"Most dense VZ sequence: {most_dense_vz['name']} (density: {most_dense_vz['density_vz']:.3f})")
    print(f"Most dense IR sequence: {most_dense_ir['name']} (density: {most_dense_ir['density_ir']:.3f})")
    print(f"Least dense VZ sequence: {least_dense_vz['name']} (density: {least_dense_vz['density_vz']:.3f})")
    print(f"Least dense IR sequence: {least_dense_ir['name']} (density: {least_dense_ir['density_ir']:.3f})")
    print()

    # Distribution analysis
    print("=== DISTRIBUTION ANALYSIS ===")
    density_ranges_vz = {'very_low': 0, 'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
    density_ranges_ir = {'very_low': 0, 'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
    
    for seq in sequence_stats:
        # VZ density categorization
        if seq['density_vz'] == 0:
            density_ranges_vz['very_low'] += 1
        elif seq['density_vz'] < 0.1:
            density_ranges_vz['low'] += 1
        elif seq['density_vz'] < 0.5:
            density_ranges_vz['medium'] += 1
        elif seq['density_vz'] < 1.0:
            density_ranges_vz['high'] += 1
        else:
            density_ranges_vz['very_high'] += 1
            
        # IR density categorization
        if seq['density_ir'] == 0:
            density_ranges_ir['very_low'] += 1
        elif seq['density_ir'] < 0.1:
            density_ranges_ir['low'] += 1
        elif seq['density_ir'] < 0.5:
            density_ranges_ir['medium'] += 1
        elif seq['density_ir'] < 1.0:
            density_ranges_ir['high'] += 1
        else:
            density_ranges_ir['very_high'] += 1
    
    print("VZ Density Distribution:")
    for category, count in density_ranges_vz.items():
        print(f"  {category.replace('_', ' ').title()}: {count} sequences ({count/len(sequence_stats)*100:.1f}%)")
    
    print("IR Density Distribution:")
    for category, count in density_ranges_ir.items():
        print(f"  {category.replace('_', ' ').title()}: {count} sequences ({count/len(sequence_stats)*100:.1f}%)")
    print()

    # Data quality insights
    print("=== DATA QUALITY INSIGHTS ===")
    zero_annotation_sequences_vz = sum(1 for seq in sequence_stats if seq['annotations_vz'] == 0)
    zero_annotation_sequences_ir = sum(1 for seq in sequence_stats if seq['annotations_ir'] == 0)
    
    print(f"Sequences with zero VZ annotations: {zero_annotation_sequences_vz} ({zero_annotation_sequences_vz/len(sequence_stats)*100:.1f}%)")
    print(f"Sequences with zero IR annotations: {zero_annotation_sequences_ir} ({zero_annotation_sequences_ir/len(sequence_stats)*100:.1f}%)")
    
    # Balance analysis
    vz_ir_ratio = total_frames_vz / total_frames_ir if total_frames_ir > 0 else float('inf')
    print(f"VZ to IR frames ratio: {vz_ir_ratio:.2f}:1")
    
    annotation_ratio = total_amount_annotations_vz / total_amount_annotations_ir if total_amount_annotations_ir > 0 else float('inf')
    print(f"VZ to IR annotations ratio: {annotation_ratio:.2f}:1")