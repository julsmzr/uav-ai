import os
import cv2
from tqdm import tqdm
import json
import shutil

from uav.setup.utils import vprint


def extract_frames(source_mp4: str, sequence_dirname: str, modality_str: str, target_dir: str) -> tuple[int, tuple[int, int]]:
    """Extract the frames from the mp4 video and create jpg files at the target directory. Returns the amount of frames extracted."""
    cap = cv2.VideoCapture(source_mp4)
    frame_count = 0
    success = True

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while success:
        success, frame = cap.read()
        if not success:
            break

        image_name = f"{sequence_dirname}-{modality_str}-{frame_count:08d}.jpg"
        cv2.imwrite(os.path.join(target_dir, image_name), frame)
        frame_count += 1

    cap.release()
    return frame_count, (width, height)

def extract_labels(source_json: str, sequence_dirname: str, modality_str: str, target_dir: str, resolution: tuple[int, int]) -> int:
    """Extract the labels from the json file and create txt files at the target directory. Returns the amount of annotated labels extracted."""
    with open(source_json, "r") as f:
        data = json.load(f)
    
    img_width, img_height = resolution
    
    for idx, (exists, bbox) in enumerate(zip(data["exist"], data["gt_rect"])):
        label_name = f"{sequence_dirname}-{modality_str}-{idx:08d}.txt"
        
        if exists and bbox and len(bbox) == 4:
            x, y, w, h = bbox[:4]
            
            # Normalize coordinates
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            
            # Ensure coordinates are within [0, 1] range
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))
            
            label_content = " ".join(["0", f"{x_center:.6f}", f"{y_center:.6f}", f"{w_norm:.6f}", f"{h_norm:.6f}"])
        else:
            label_content = ""

        with open(os.path.join(target_dir, label_name), "w") as f:
            f.write(label_content)

    return sum(1 for exists in data["exist"] if exists)

def get_target_sequence_count(source_dir: str) -> int:
    """Count all sequences in source directory."""
    amount_sequences = 0
    for subset_dirname in ["test", "train", "val"]:
        amount_sequences += len(os.listdir(os.path.join(source_dir, subset_dirname)))
    return amount_sequences


def remove_sourcedir(source_dir: str, verbose: bool) -> None:
    """Remove the source directory if it exists."""
    if os.path.exists(source_dir):
        try:
            shutil.rmtree(source_dir)
            vprint(verbose, "Source directory was removed successfully.")
        except OSError as e:
            vprint(verbose, f"Warning: Could not remove source directory {source_dir}: {e}")

def validate_existing_sequences(source_dir: str, target_imagesdir: str, target_labelsdir: str, verbose: bool) -> tuple[dict, list[str]]:
    """Check if sequences have already been extracted by comparing file counts. Returns statistics dict and list of validated sequence names."""
    validated_sequences = []
    statistics = {}
    if not os.path.exists(target_imagesdir) or not os.path.exists(target_labelsdir):
        return statistics, validated_sequences
    
    for subset_dirname in ["test", "train", "val"]:
        subset_dir = os.path.join(source_dir, subset_dirname)
        
        if not os.path.exists(subset_dir):
            continue
            
        for sequence_dirname in tqdm(os.listdir(subset_dir), desc=f"Validating {subset_dirname} subset"):
            if sequence_dirname == ".DS_Store":
                continue
                
            sequence_image_dir = os.path.join(target_imagesdir, sequence_dirname)
            sequence_label_dir = os.path.join(target_labelsdir, sequence_dirname)
            
            if not os.path.exists(sequence_image_dir) or not os.path.exists(sequence_label_dir):
                continue
            
            # Get expected frame counts from video files
            visible_video_path = os.path.join(subset_dir, sequence_dirname, "visible.mp4")
            infrared_video_path = os.path.join(subset_dir, sequence_dirname, "infrared.mp4")
            
            cap_vz = cv2.VideoCapture(visible_video_path)
            expected_frames_vz = int(cap_vz.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_vz.release()
            
            cap_ir = cv2.VideoCapture(infrared_video_path)
            expected_frames_ir = int(cap_ir.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_ir.release()
            
            # Count actual amount of files
            actual_images = len([f for f in os.listdir(sequence_image_dir) if f.endswith('.jpg')])
            actual_labels = len([f for f in os.listdir(sequence_label_dir) if f.endswith('.txt')])
            
            expected_total_images = expected_frames_vz + expected_frames_ir
            expected_total_labels = expected_frames_vz + expected_frames_ir  # Same as frames
            
            if actual_images == expected_total_images and actual_labels == expected_total_labels:
                validated_sequences.append(sequence_dirname)
                with open(os.path.join(subset_dir, sequence_dirname, "visible.json"), "r") as f:
                    data_vz = json.load(f)
                amount_annotations_vz = sum(1 for exists in data_vz["exist"] if exists)
                
                with open(os.path.join(subset_dir, sequence_dirname, "infrared.json"), "r") as f:
                    data_ir = json.load(f)
                amount_annotations_ir = sum(1 for exists in data_ir["exist"] if exists)
                
                statistics[sequence_dirname] = (expected_frames_vz, expected_frames_ir, amount_annotations_vz, amount_annotations_ir)
    
    amount_validated_sequences = len(validated_sequences)
    target_amount_sequences = get_target_sequence_count(source_dir=source_dir)

    if amount_validated_sequences == target_amount_sequences:
        vprint(verbose, "The entire dataset was already extracted and is valid.")
        remove_sourcedir(source_dir, verbose)
        exit(0)

    vprint(verbose, f"Sucessfully validated {amount_validated_sequences} sequences.")
    vprint(verbose, f"Continuing extraction for {target_amount_sequences - amount_validated_sequences} remaining sequences...")
    return statistics, validated_sequences

def format_dataset(source_dir: str, target_dir: str, verbose: bool, remove_source: bool) -> None:
    """Converts Anti-UAV300 raw MP4 videos and JSON annotations to YOLO format."""
    target_imagesdir = os.path.join(target_dir, "images")
    target_labelsdir = os.path.join(target_dir, "labels")

    if os.path.exists(target_dir):
        vprint(verbose, "Detected Target Directory. Validating...\n")
        statistics, validated = validate_existing_sequences(source_dir, target_imagesdir, target_labelsdir, verbose)
    else:
        vprint(verbose, "Starting Extraction of Data...\n")
        os.makedirs(target_dir)
        statistics = {} # sequence_name(str) : (amount_frames_vz, amount_frames_ir, amount_annotations_vz, amount_annotations_ir)(tuple[int, int, int, int])
        validated = None

    for subset_dirname in ["test", "train", "val"]:
        subset_dir = os.path.join(source_dir, subset_dirname)
        for sequence_dirname in tqdm(os.listdir(subset_dir), desc=f"Extracting {subset_dirname} subset"):
            if sequence_dirname == ".DS_Store" or (sequence_dirname in validated if validated is not None else False):
                continue

            # Extract frames
            target_image_sequencedir = os.path.join(target_imagesdir, sequence_dirname)
            if os.path.exists(target_image_sequencedir): 
                shutil.rmtree(target_image_sequencedir) # when it was only partially extracted
            os.makedirs(target_image_sequencedir)

            amount_frames_vz, (video_width_vz, video_height_vz) = extract_frames(source_mp4=os.path.join(subset_dir, sequence_dirname, "visible.mp4"), sequence_dirname=sequence_dirname, modality_str="vz", target_dir=os.path.join(target_imagesdir, sequence_dirname))
            amount_frames_ir, (video_width_ir, video_height_ir) = extract_frames(source_mp4=os.path.join(subset_dir, sequence_dirname, "infrared.mp4"), sequence_dirname=sequence_dirname, modality_str="ir", target_dir=os.path.join(target_imagesdir, sequence_dirname))

            # Extract labels
            target_label_sequencedir = os.path.join(target_labelsdir, sequence_dirname)
            if os.path.exists(target_label_sequencedir): 
                shutil.rmtree(target_label_sequencedir) # when it was only partially extracted
            os.makedirs(target_label_sequencedir)

            amount_labels_vz = extract_labels(source_json=os.path.join(subset_dir, sequence_dirname, "visible.json"), sequence_dirname=sequence_dirname, modality_str="vz", target_dir=os.path.join(target_labelsdir, sequence_dirname), resolution=(video_width_vz, video_height_vz) )
            amount_labels_ir = extract_labels(source_json=os.path.join(subset_dir, sequence_dirname, "infrared.json"), sequence_dirname=sequence_dirname, modality_str="ir", target_dir=os.path.join(target_labelsdir, sequence_dirname), resolution=(video_width_ir, video_height_ir))

            statistics[sequence_dirname] = (amount_frames_vz, amount_frames_ir, amount_labels_vz, amount_labels_ir)

            if remove_source:
                sequence_source_dir = os.path.join(subset_dir, sequence_dirname)
                if os.path.exists(sequence_source_dir):
                    shutil.rmtree(sequence_source_dir)

    statistics_filepath = os.path.join(target_dir, "statistics.json")
    with open(statistics_filepath, "w") as f:
        json.dump(statistics, f, indent=4)
    vprint(verbose, f"Statistics successfully saved to {statistics_filepath}.")

    if remove_source:
        remove_sourcedir(source_dir, verbose)

    vprint(verbose, "Conversion done!")

def process_dataset(source_dir: str, target_dir: str, verbose: bool, remove_source: bool) -> bool:
    """Processes dataset with error handling, returns True on success."""
    print(source_dir)
    if not os.path.exists(source_dir):
        raise RuntimeError(f"Source directory '{source_dir}' does not exist.")
    
    try:
        format_dataset(source_dir, target_dir, verbose, remove_source)
    except KeyboardInterrupt:
        print("Data processing aborted.")
        return False
    
    return True
