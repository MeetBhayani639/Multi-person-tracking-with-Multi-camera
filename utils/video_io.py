"""
utils/video_io.py

Helpers to create a stable OpenCV VideoWriter (XVID/AVI) and optionally convert AVI -> MP4 using ffmpeg.
"""
import os
import subprocess
import cv2


def create_writer(out_path_mp4: str, fps: float, frame_size: tuple):
    """
    Returns (out_path, cv2.VideoWriter) where out_path is an AVI path (we enforce .avi).
    Caller should write until done and release the writer.

    Parameters:
      - out_path_mp4: suggested output path ending with .mp4 (the old API), we replace ext with .avi
      - fps: frames per second
      - frame_size: (width, height)
    """
    base, ext = os.path.splitext(out_path_mp4)
    out_path_avi = base + ".avi"

    # Choose XVID: works well cross-platform and produces readable AVI files
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(out_path_avi, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for path: {out_path_avi}")
    return out_path_avi, writer


def ffmpeg_exists():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def convert_avi_to_mp4(avi_path: str, mp4_path: str, crf: int = 23):
    """
    Convert AVI -> MP4 (H.264 + faststart) using ffmpeg if available.
    Returns True if conversion successful, False otherwise.
    """
    if not ffmpeg_exists():
        return False

    # Build ffmpeg command
    # -movflags +faststart helps web playback (optimized for streaming)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", avi_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        mp4_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False
