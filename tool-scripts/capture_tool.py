import subprocess
import os
import time
import argparse

def capture_frame(filepath):
    cmd = (
        "gst-launch-1.0 v4l2src device=/dev/video-imx219-cam0 num-buffers=3 ! "
        "video/x-bayer,width=1920,height=1080,format=rggb ! "
        "bayer2rgb ! videoconvert ! video/x-raw,format=BGR ! "
        "videoconvert ! jpegenc ! "
        f"filesink location={filepath}"
    )
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def start_preview():
    cmd = (
        "gst-launch-1.0 v4l2src device=/dev/video-imx219-cam0 ! "
        "video/x-bayer,width=1920,height=1080,format=rggb ! "
        "bayer2rgb ! videoconvert ! video/x-raw,format=BGR ! "
        "videoconvert ! kmssink sync=false"
    )
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='/home/root/captures/tool')
    parser.add_argument('--prefix', type=str, default='frame')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    saved_count = len([f for f in os.listdir(args.output) if f.endswith('.jpg')])
    print(f"Saving to: {args.output} | Already have: {saved_count} images")
    print("ENTER = capture | Q = quit")

    preview = start_preview()
    time.sleep(2)
    print("Preview started on HDMI!")

    try:
        while True:
            user_input = input(f"[{saved_count} saved] ENTER=capture, Q=quit: ").strip().lower()
            if user_input == 'q':
                print(f"Done! Saved {saved_count} images.")
                break
            preview.terminate()
            time.sleep(0.5)
            filepath = os.path.join(args.output, f"{args.prefix}{saved_count:04d}.jpg")
            capture_frame(filepath)
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                saved_count += 1
                print(f"  Saved! ({saved_count} total)")
            else:
                print(f"  Failed, try again")
            preview = start_preview()
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"Interrupted. Saved {saved_count} images.")
    finally:
        preview.terminate()

if __name__ == '__main__':
    main()
