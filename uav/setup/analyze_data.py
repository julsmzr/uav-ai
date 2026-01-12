import json

def analyze_data(stats_path: str = "datasets/anti-uav300/statistics.json") -> None:
    """Analyzes and prints Anti-UAV300 dataset statistics from JSON file."""
    with open(stats_path, "r") as f:
        data = json.load(f)

    total_sequences = len(data)
    frames_vz = sum(seq[0] for seq in data.values())
    frames_ir = sum(seq[1] for seq in data.values())
    annot_vz = sum(seq[2] for seq in data.values())
    annot_ir = sum(seq[3] for seq in data.values())

    ratio_vz = annot_vz / frames_vz
    ratio_ir = annot_ir / frames_ir

    print(f"\n{'='*50}")
    print(f"Anti-UAV300 Dataset Statistics")
    print(f"{'='*50}\n")

    print(f"Total Sequences: {total_sequences}\n")

    print(f"Visible Spectrum:")
    print(f"  Total Frames:       {frames_vz:,}")
    print(f"  Annotated Frames:   {annot_vz:,}")
    print(f"  Coverage:           {ratio_vz:.2%}\n")

    print(f"Infrared:")
    print(f"  Total Frames:       {frames_ir:,}")
    print(f"  Annotated Frames:   {annot_ir:,}")
    print(f"  Coverage:           {ratio_ir:.2%}\n")

    print(f"Total Dataset:")
    print(f"  Total Frames:       {frames_vz + frames_ir:,}")
    print(f"  Annotated Frames:   {annot_vz + annot_ir:,}")
    print(f"  Overall Coverage:   {(annot_vz + annot_ir) / (frames_vz + frames_ir):.2%}\n")
    print(f"{'='*50}\n")
