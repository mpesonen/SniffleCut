"""
FCPXML script to automatically cut out detected sniffles from asset-clips.
Processes FCPXML files, detects sniffles in audio, and modifies the XML to skip sniffle segments.
"""

import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from fractions import Fraction
from urllib.parse import unquote, urlparse

# Configuration
EPS = 0.10  # Pre/post cut duration in seconds (padding around sniffles)
THRESHOLD = 0.8  # Probability threshold for sniffle detection

# Get script directory and project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR
SNIFFLE_DETECT_PATH = PROJECT_ROOT / "sniffle_detect_and_train.py"
SNIFFLE_HEAD_PATH = PROJECT_ROOT / "training_checkpoints" / "sniffle_head_4epochs.pt"


def parse_timecode(timecode_str):
    """Parse FCPXML timecode string (e.g., '3065366000/50000s') to seconds."""
    if not timecode_str or timecode_str == "0s":
        return 0.0
    
    if timecode_str.endswith("s"):
        timecode_str = timecode_str[:-1]
    
    if "/" in timecode_str:
        parts = timecode_str.split("/")
        numerator = Fraction(parts[0])
        denominator = Fraction(parts[1]) if len(parts) > 1 else Fraction(1)
        return float(numerator / denominator)
    else:
        return float(timecode_str)


def format_timecode(seconds, denominator=50000):
    """Format seconds to FCPXML timecode format."""
    numerator = int(seconds * denominator)
    return f"{numerator}/{denominator}s"


def file_url_to_path(file_url):
    """Convert file:// URL to filesystem path."""
    parsed = urlparse(file_url)
    path = unquote(parsed.path)
    # Remove leading slash on macOS/Unix
    if path.startswith("/") and sys.platform != "win32":
        return path
    return path


def extract_audio_from_video(video_path, start_sec, duration_sec, output_path):
    """Extract audio from video file using ffmpeg."""
    # Validate inputs
    if duration_sec <= 0:
        raise ValueError(f"Invalid duration: {duration_sec} seconds")
    if start_sec < 0:
        raise ValueError(f"Invalid start time: {start_sec} seconds")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-vn",  # No video
        "-ac", "1",  # Mono
        "-ar", "16000",  # 16kHz sample rate
        "-y",  # Overwrite output file
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode() if e.stderr else str(e)
        # Check if error is about missing audio stream
        if "Stream map '0:a?' matches no streams" in stderr_msg or "No audio streams" in stderr_msg:
            raise RuntimeError(
                f"Video file has no audio track: {video_path}\n"
                f"Cannot extract audio for sniffle detection."
            )
        raise RuntimeError(
            f"Failed to extract audio with ffmpeg: {stderr_msg}\n"
            f"Command: {' '.join(cmd)}"
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg.\n"
            "Install with: brew install ffmpeg (macOS)"
        )
    
    # Verify output file was created and has content
    output_path_obj = Path(output_path)
    if not output_path_obj.exists():
        raise RuntimeError(f"Audio extraction failed: output file not created: {output_path}")
    
    file_size = output_path_obj.stat().st_size
    if file_size == 0:
        raise RuntimeError(
            f"Audio extraction failed: output file is empty (0 bytes)\n"
            f"This might indicate the video has no audio track or the time range is invalid.\n"
            f"Video: {video_path}\n"
            f"Start: {start_sec}s, Duration: {duration_sec}s"
        )
    
    # Check minimum file size (very small files are likely invalid)
    min_size = 1000  # 1KB minimum for valid WAV header + some data
    if file_size < min_size:
        print(f"Warning: Extracted audio file is very small ({file_size} bytes). This might indicate an issue.")
    
    return output_path_obj


def detect_sniffles(wav_path, threshold=THRESHOLD):
    """
    Detect sniffles in audio file using the BEATs model.
    Returns a list of dictionaries with 'start', 'end', and 'probability' keys.
    """
    python_bin = PROJECT_ROOT / ".venv" / "bin" / "python"
    cmd = [
        str(python_bin),
        str(SNIFFLE_DETECT_PATH),
        "--mode", "test",
        "--wav", str(wav_path),
        "--threshold", str(threshold),
        "--head_checkpoint", str(SNIFFLE_HEAD_PATH),
    ]
    
    try:
        result = subprocess.check_output(
            cmd,
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.PIPE,
            text=True
        )
        sniffles = json.loads(result.strip())
        return sniffles
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to detect sniffles: {e.stderr if e.stderr else str(e)}\n"
            f"Command: {' '.join(cmd)}"
        )
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse detection results: {e}")


def find_asset_clips(root):
    """Find all asset-clip elements in the FCPXML."""
    # Find all asset-clips in spine elements
    asset_clips = []
    for spine in root.findall(".//spine"):
        for clip in spine.findall("asset-clip"):
            asset_clips.append(clip)
    return asset_clips


def get_asset_by_ref(root, ref_id):
    """Find asset element by id reference."""
    for asset in root.findall(".//asset"):
        if asset.get("id") == ref_id:
            return asset
    return None


def get_media_file_path(asset):
    """Extract media file path from asset element."""
    media_rep = asset.find("media-rep")
    if media_rep is None:
        return None
    
    src = media_rep.get("src")
    if not src:
        return None
    
    return file_url_to_path(src)


def process_asset_clip(clip, assets_dict, temp_dir, threshold=THRESHOLD, eps=EPS):
    """Process a single asset-clip: detect sniffles and return cut points."""
    ref_id = clip.get("ref")
    if not ref_id:
        return None
    
    asset = assets_dict.get(ref_id)
    if asset is None:
        print(f"Warning: Asset {ref_id} not found")
        return None
    
    # Get media file path
    media_path = get_media_file_path(asset)
    if not media_path or not os.path.exists(media_path):
        print(f"Warning: Media file not found: {media_path}")
        return None
    
    # Parse clip timing
    clip_start_str = clip.get("start", "0s")
    clip_duration_str = clip.get("duration", "0s")
    clip_offset_str = clip.get("offset", "0s")
    
    clip_start_sec = parse_timecode(clip_start_str)
    clip_duration_sec = parse_timecode(clip_duration_str)
    clip_offset_sec = parse_timecode(clip_offset_str)
    
    # Get asset timing
    asset_start_str = asset.get("start", "0s")
    asset_start_sec = parse_timecode(asset_start_str)
    
    # In FCPXML, the asset's "start" attribute defines the base offset in the
    # source file where the asset begins. The asset-clip's "start" attribute
    # is the absolute timecode position in the source file.
    #
    # To extract audio, we need the position relative to the source file's beginning.
    # We calculate this by subtracting the asset's base offset from the clip's start:
    # source_start = clip_start - asset_start
    #
    # This gives us the offset within the source file where we should start extracting.
    clip_offset_within_asset = clip_start_sec - asset_start_sec
    source_start_sec = clip_offset_within_asset

    print(f"  Processing clip: {clip.get('name', 'unnamed')}")
    #print(f"    Source file: {media_path}")
    
    # Extract audio
    temp_wav = temp_dir / f"clip_{ref_id}.wav"
    try:
        extract_audio_from_video(media_path, source_start_sec, clip_duration_sec, temp_wav)
    except (RuntimeError, ValueError) as e:
        print(f"    Error extracting audio: {e}")
        return None
    
    # Validate WAV file can be read
    try:
        import torchaudio
        test_wav, test_sr = torchaudio.load(str(temp_wav))
        if test_wav.numel() == 0:
            print(f"    Warning: Extracted audio file is empty, skipping")
            return None
        print(f"    Audio extracted: {test_wav.shape[1] / test_sr:.2f}s at {test_sr}Hz")
    except Exception as e:
        print(f"    Error: Extracted audio file is invalid or corrupted: {e}")
        return None
    
    # Detect sniffles
    print(f"    Detecting sniffles...")
    sniffles = detect_sniffles(temp_wav, threshold=threshold)
    
    if not sniffles:
        print(f"    No sniffles detected")
        return None
    
    print(f"    Found {len(sniffles)} sniffle segment(s):")
    for sniffle in sniffles:
        print(f"      {sniffle['start']:.3f}s - {sniffle['end']:.3f}s (prob: {sniffle['probability']:.2f})")
    
    # Convert sniffle times from clip-relative to source-relative
    # Sniffles are relative to the extracted audio (which starts at clip_start_sec in source)
    cut_points = []
    for sniffle in sniffles:
        sniffle_start = sniffle["start"]
        sniffle_end = sniffle["end"]
        
        # Add padding
        cut_start = max(0, sniffle_start - eps)
        cut_end = min(clip_duration_sec, sniffle_end + eps)
        
        # Convert to absolute time in source file
        abs_cut_start = source_start_sec + cut_start
        abs_cut_end = source_start_sec + cut_end
        
        # Convert back to clip-relative for XML modification
        clip_relative_start = cut_start
        clip_relative_end = cut_end
        
        cut_points.append({
            "clip_start": clip_relative_start,
            "clip_end": clip_relative_end,
            "abs_start": abs_cut_start,
            "abs_end": abs_cut_end,
        })
    
    return cut_points


def split_asset_clip(clip, cut_points):
    """
    Split an asset-clip into multiple clips, removing sniffle segments.
    Returns list of new clip elements.
    """
    ref_id = clip.get("ref")
    clip_start_str = clip.get("start", "0s")
    clip_duration_str = clip.get("duration", "0s")
    
    clip_start_sec = parse_timecode(clip_start_str)
    clip_duration_sec = parse_timecode(clip_duration_str)
    
    # Sort cut points by start time
    cut_points = sorted(cut_points, key=lambda x: x["clip_start"])
    
    # Build segments to keep (gaps between sniffles)
    segments = []
    current_pos = 0.0
    
    for cut_point in cut_points:
        cut_start = cut_point["clip_start"]
        cut_end = cut_point["clip_end"]
        
        # Add segment before this cut if there's a gap
        if cut_start > current_pos:
            segments.append({
                "start": current_pos,
                "end": cut_start,
            })
        
        current_pos = max(current_pos, cut_end)
    
    # Add final segment if there's remaining content
    if current_pos < clip_duration_sec:
        segments.append({
            "start": current_pos,
            "end": clip_duration_sec,
        })
    
    # If no segments (entire clip is sniffles), return empty list
    if not segments:
        return []
    
    # Create new clips for each segment
    new_clips = []
    for segment in segments:
        # Create new asset-clip element
        new_clip = ET.Element("asset-clip")
        
        # Copy attributes from original
        for key, value in clip.attrib.items():
            if key not in ["start", "duration", "offset"]:
                new_clip.set(key, value)
        
        # Set new timing
        segment_start_sec = clip_start_sec + segment["start"]
        segment_duration_sec = segment["end"] - segment["start"]
        
        # Use same denominator as original for consistency
        denominator = 50000  # Default, could parse from original
        if "/" in clip_start_str:
            denominator = int(clip_start_str.split("/")[1].rstrip("s"))
        
        new_clip.set("start", format_timecode(segment_start_sec, denominator))
        new_clip.set("duration", format_timecode(segment_duration_sec, denominator))
        
        # Offset will be recalculated after all clips are inserted
        # For now, set a placeholder - we'll fix it later
        new_clip.set("offset", "0s")
        
        # Copy child elements (like conform-rate)
        for child in clip:
            new_clip.append(ET.fromstring(ET.tostring(child)))
        
        new_clips.append(new_clip)
    
    return new_clips


def process_fcpxml(input_path, output_path=None, threshold=THRESHOLD, eps=EPS):
    """Process FCPXML file to cut out sniffles."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"FCPXML file not found: {input_path}")
    
    if output_path is None:
        # Create output path in a new folder with "_no_sniffles" suffix
        # Preserve the bundle structure (e.g., FCPXML.fcpxmld/Info.fcpxml -> FCPXML_no_sniffles.fcpxmld/Info.fcpxml)
        input_path = Path(input_path)
        parent_dir = input_path.parent
        parent_name = parent_dir.name
        
        # If parent is a .fcpxmld bundle, create new bundle with suffix
        if parent_dir.suffix == ".fcpxmld" or parent_name.endswith(".fcpxmld"):
            new_parent_name = parent_name.replace(".fcpxmld", "_no_sniffles.fcpxmld")
            if not new_parent_name.endswith(".fcpxmld"):
                new_parent_name = f"{parent_name}_no_sniffles.fcpxmld"
            output_dir = parent_dir.parent / new_parent_name
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / input_path.name  # Keep Info.fcpxml name
        else:
            # If not a bundle, just add suffix to parent
            output_dir = parent_dir.parent / f"{parent_name}_no_sniffles"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / input_path.name
    else:
        output_path = Path(output_path)
    
    print(f"Processing FCPXML: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Parse XML
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    # Build assets dictionary for quick lookup
    assets_dict = {}
    for asset in root.findall(".//asset"):
        asset_id = asset.get("id")
        if asset_id:
            assets_dict[asset_id] = asset
    
    # Find all asset-clips
    asset_clips = find_asset_clips(root)
    print(f"\nFound {len(asset_clips)} asset-clip(s) to process")
    
    if not asset_clips:
        print("No asset-clips found in FCPXML")
        return
    
    # Process each clip
    temp_dir = Path(tempfile.mkdtemp(prefix="sniffle_cut_"))
    clips_to_replace = {}  # Maps original clip element to new clips
    
    try:
        for i, clip in enumerate(asset_clips, 1):
            print(f"\n[{i}/{len(asset_clips)}] Processing asset-clip...")
            
            cut_points = process_asset_clip(clip, assets_dict, temp_dir, threshold=threshold, eps=eps)
            
            if cut_points:
                new_clips = split_asset_clip(clip, cut_points)
                if new_clips:
                    clips_to_replace[clip] = new_clips
                    print(f"    Will split into {len(new_clips)} segment(s)")
                else:
                    print(f"    Warning: Entire clip would be removed (all sniffles)")
            else:
                print(f"    No changes needed")
    
    finally:
        # Clean up temp files
        for temp_file in temp_dir.glob("*.wav"):
            try:
                temp_file.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete temp file {temp_file}: {e}")
        try:
            temp_dir.rmdir()
        except Exception:
            pass
    
    # Replace clips in XML
    if clips_to_replace:
        print(f"\nModifying FCPXML to remove sniffle segments...")
        for original_clip, new_clips in clips_to_replace.items():
            # Find parent by searching for the clip in all spine elements
            parent = None
            for spine in root.findall(".//spine"):
                if original_clip in list(spine):
                    parent = spine
                    break
            
            if parent is None:
                print(f"Warning: Could not find parent for clip, skipping")
                continue
            
            # Find position of original clip
            index = list(parent).index(original_clip)
            
            # Remove original
            parent.remove(original_clip)
            
            # Insert new clips at same position (in reverse to maintain order)
            for j, new_clip in enumerate(new_clips):
                parent.insert(index + j, new_clip)
        
        print(f"Modified {len(clips_to_replace)} clip(s)")
        
        # Recalculate all offsets to ensure continuity (no gaps)
        print("Recalculating offsets for continuity...")
        for spine in root.findall(".//spine"):
            clips = list(spine)
            if not clips:
                continue
            
            # Get denominator from first clip (or use default)
            denominator = 50000
            if clips[0].get("offset"):
                offset_str = clips[0].get("offset", "0s")
                if "/" in offset_str:
                    try:
                        denominator = int(offset_str.split("/")[1].rstrip("s"))
                    except (ValueError, IndexError):
                        pass
            
            # Recalculate offsets sequentially
            current_offset_sec = 0.0
            for clip in clips:
                # Set offset for this clip
                clip.set("offset", format_timecode(current_offset_sec, denominator))
                
                # Calculate next offset by adding this clip's duration
                duration_str = clip.get("duration", "0s")
                duration_sec = parse_timecode(duration_str)
                current_offset_sec += duration_sec
    else:
        print("\nNo modifications needed - no sniffles found or all clips would be removed")
        return
    
    # Write modified XML
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    print(f"\n✓ Saved modified FCPXML to: {output_path}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cut sniffles from FCPXML asset-clips using BEATs detection."
    )
    parser.add_argument(
        "fcpxml",
        type=str,
        help="Path to input FCPXML file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output FCPXML file (default: input_name_no_sniffles.fcpxml)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Probability threshold for sniffle detection (default: {THRESHOLD})"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=EPS,
        help=f"Padding around sniffles in seconds (default: {EPS})"
    )
    
    # Print help if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    # Verify dependencies
    if not SNIFFLE_DETECT_PATH.exists():
        raise FileNotFoundError(
            f"Detection script not found: {SNIFFLE_DETECT_PATH}\n"
            "Make sure sniffle_detect_and_train.py exists."
        )
    
    if not SNIFFLE_HEAD_PATH.exists():
        raise FileNotFoundError(
            f"Sniffle head model not found: {SNIFFLE_HEAD_PATH}\n"
            "Please train the model first using sniffle_detect_and_train.py"
        )
    
    try:
        process_fcpxml(
            args.fcpxml,
            args.output,
            threshold=args.threshold,
            eps=args.eps
        )
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
