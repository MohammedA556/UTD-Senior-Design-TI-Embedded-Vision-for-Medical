#!/usr/bin/env python3
"""
Simple CSI camera recording script for the TI AM62A with IMX219.
Records video from the CSI camera and saves it as MP4.

Usage:
    python3 record_csi.py                          # Record for 30 seconds (default)
    python3 record_csi.py -d 60                    # Record for 60 seconds
    python3 record_csi.py -d 60 -o my_video.mp4    # Custom output filename
    python3 record_csi.py -r 1280x720              # Custom resolution
    python3 record_csi.py --device /dev/video2      # Custom video device

Requires: GStreamer 1.0 installed on the AM62A target board.
"""

import subprocess
import argparse
import os
import sys
from datetime import datetime


def get_pipeline(device, subdev, width, height, framerate, output_file, sensor_id="imx219"):
    """
    Build the GStreamer pipeline string for CSI camera recording on AM62A.
    """
    # Raw bayer pipeline with TI ISP for IMX219
    pipeline = (
        f'v4l2src device={device} io-mode=5 ! '
        f'queue leaky=2 ! '
        f'video/x-bayer, width={width}, height={height}, format=rggb, framerate={framerate}/1 ! '
        f'tiovxisp '
        f'sensor-name=SENSOR_SONY_IMX219_RPI '
        f'dcc-isp-file=/opt/imaging/{sensor_id}/linear/dcc_viss.bin '
        f'dcc-2a-file=/opt/imaging/{sensor_id}/linear/dcc_2a.bin '
        f'device={subdev} format-msb=7 ! '
        f'video/x-raw, format=NV12, width={width}, height={height} ! '
        f'tiovxmemalloc pool-size=4 ! '
        f'v4l2h264enc bitrate=10000000 ! '
        f'h264parse ! '
        f'mp4mux ! '
        f'filesink location={output_file}'
    )
    return pipeline


def get_raw_pipeline(device, width, height, framerate, output_file):
    """
    Fallback pipeline for non-bayer cameras (e.g. USB or MJPEG CSI).
    """
    pipeline = (
        f'v4l2src device={device} ! '
        f'video/x-raw, width={width}, height={height}, framerate={framerate}/1 ! '
        f'v4l2h264enc bitrate=10000000 ! '
        f'h264parse ! '
        f'mp4mux ! '
        f'filesink location={output_file}'
    )
    return pipeline


def record(pipeline_str, duration):
    """
    Run gst-launch-1.0 with the given pipeline for the specified duration.
    """
    cmd = f'timeout {duration} gst-launch-1.0 -e {pipeline_str}'
    print(f"Recording for {duration} seconds...")
    print(f"Pipeline: {pipeline_str}\n")
    
    try:
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        process.terminate()
        process.wait()


def main():
    parser = argparse.ArgumentParser(description="Record video from CSI camera on AM62A")
    parser.add_argument("-d", "--duration", type=int, default=30,
                        help="Recording duration in seconds (default: 30)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output filename (default: recording_TIMESTAMP.mp4)")
    parser.add_argument("-r", "--resolution", type=str, default="1920x1080",
                        help="Resolution WIDTHxHEIGHT (default: 1920x1080)")
    parser.add_argument("-f", "--framerate", type=int, default=30,
                        help="Framerate (default: 30)")
    parser.add_argument("--device", type=str, default="/dev/video-imx219-cam0",
                        help="Camera device path (default: /dev/video-imx219-cam0)")
    parser.add_argument("--subdev", type=str, default="/dev/v4l-imx219-subdev0",
                        help="Camera subdevice path (default: /dev/v4l-imx219-subdev0)")
    parser.add_argument("--sensor", type=str, default="imx219",
                        help="Sensor ID (default: imx219)")
    parser.add_argument("--raw", action="store_true",
                        help="Use raw/non-bayer pipeline (for USB or other cameras)")
    parser.add_argument("--output-dir", type=str, default="/opt/edgeai-test-data/output",
                        help="Output directory (default: /opt/edgeai-test-data/output)")
    
    args = parser.parse_args()
    
    width, height = args.resolution.split("x")
    width, height = int(width), int(height)
    
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"recording_{timestamp}.mp4"
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output)
    
    if args.raw:
        pipeline = get_raw_pipeline(args.device, width, height, args.framerate, output_path)
    else:
        pipeline = get_pipeline(args.device, args.subdev, width, height, 
                               args.framerate, output_path, args.sensor)
    
    print("=" * 50)
    print("  AM62A CSI Camera Recorder")
    print("=" * 50)
    print(f"  Device:     {args.device}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Framerate:  {args.framerate} fps")
    print(f"  Duration:   {args.duration} seconds")
    print(f"  Output:     {output_path}")
    print("=" * 50)
    print("Press Ctrl+C to stop early.\n")
    
    record(pipeline, args.duration)
    
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nRecording saved: {output_path} ({size_mb:.1f} MB)")
    else:
        print("\nWarning: Output file was not created. Check camera connection and device path.")


if __name__ == "__main__":
    main()
