#!/usr/bin/python3
#  Copyright (C) 2021 Texas Instruments Incorporated - http://www.ti.com/
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#    Neither the name of Texas Instruments Incorporated nor the names of
#    its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import yaml
import cv2            
import numpy as np    
import os                      
import subprocess     
import select  
import struct
import time
import importlib        
import config_parser
from datetime import datetime   

from edge_ai_class import EdgeAIDemo
import utils

def show_final_summary(last_seen, class_colors, start_time):
    if not last_seen:
        return 

    # 1. Create the Canvas (Existing logic)
    height, width = 1080, 1920 # Matching standard 720p output
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30) 

    cv2.putText(canvas, "SESSION SUMMARY: FINAL DETECTIONS", (40, 80), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

    max_time = max(last_seen.values()) if last_seen else 1.0
    timeline_y, margin_x, timeline_w = 180, 80, width - 160
    cv2.line(canvas, (margin_x, timeline_y), (margin_x + timeline_w, timeline_y), (150, 150, 150), 2)

    sorted_items = sorted(last_seen.items(), key=lambda x: x[1], reverse=True)
    for i, (cls_tuple, t) in enumerate(sorted_items[:10]): # Show top 10
        cls, count = cls_tuple
        color = class_colors.get(cls, (0, 255, 0))
        y = 300 + (i * 45)
        line_y = timeline_y + 20 + (i*5)
        x_pos = margin_x + int((t / max_time) * timeline_w)
        abs_timestamp = start_time + t 
        time_str = datetime.fromtimestamp(abs_timestamp).strftime('%H:%M:%S')
        cv2.circle(canvas, (x_pos, timeline_y), 10, color, -1)
        cv2.line(canvas, (x_pos, timeline_y), (x_pos, line_y), color, 1) 
        cv2.putText(canvas, f"{i}", (x_pos, line_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        text = f"{i}. {cls}, C={count}: Last seen at t={t:.2f}, Timestamp: {time_str}"
        cv2.putText(canvas, text, (margin_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 2. Save a temporary file for GStreamer to read
    temp_path = "/tmp/final_summary.jpg"
    cv2.imwrite(temp_path, canvas)

    # 3. Launch GStreamer to display the image on the hardware sink
    # We use imagefreeze to turn the jpg into a continuous stream for kmssink
    gst_cmd = [
        "gst-launch-1.0", "filesrc", f"location={temp_path}", "!", 
        "jpegdec", "!", "imagefreeze", "!", "videoconvert", "!", 
        "video/x-raw,format=NV12", "!", "kmssink", "sync=false"
    ]
    
    # Start the display process in the background
    proc = subprocess.Popen(gst_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    time.sleep(1)
    print(f"\n\n" + "="*60)
    print("SHOWING SUMMARY ON SCREEN...")
    print("Program will end in 60 seconds OR press ENTER to exit now.")
    print("="*60 + "\n")

    # 4. Wait for 5 seconds OR a key press in the terminal
    try:
        # select.select monitors stdin for 5 seconds
        i, o, e = select.select([sys.stdin], [], [], 60)
        if i:
            sys.stdin.readline() # Clear the keypress from buffer
            print("Terminating via keypress...")
    except KeyboardInterrupt:
        print("Terminating via interrupt...")
    finally:
        # Kill the GStreamer display process
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("Session Closed.")


def main(sys_argv):
    args = utils.get_cmdline_args(sys_argv)

    # Linux Input Event Format for 64-bit AM62A (2 longs, 2 shorts, 1 int = 24 bytes)
    EVENT_FORMAT = "llHHi" 
    EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
    EV_KEY = 1
    BTN_LEFT = 272

    keep_cycling = True

    while keep_cycling:
        demo = None # <-- FIX 2: Initialize to None so the finally block doesn't crash
        try:
            # --- FIX 1: Hard reset TI's static counters and lists ---
            importlib.reload(config_parser)
            utils.report_list.clear()

            # Load the config fresh every single cycle
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
            # --------------------------------------------------------

            print("\n[INFO] Starting/Restarting Demo Cycle...")
            demo = EdgeAIDemo(config)
            demo.start()

            if args.verbose:
                utils.print_stdout = True

            if not args.no_curses:
                utils.enable_curses_reports(demo.title)

            # =========================================================
            # --- MINIMALLY INVASIVE REPLACEMENT FOR wait_for_exit() ---
            # =========================================================
            try:
                mouse_fd = open("/dev/input/event0", "rb")
                os.set_blocking(mouse_fd.fileno(), False)
            except Exception as e:
                mouse_fd = None

            click_start_time = None
            HOLD_THRESHOLD = 1.0 # 1 Second hold to end cycle

            while True:
                # 1. Check if demo ended internally (e.g. video finished)
                if all(i.stop_thread for i in demo.infer_pipes):
                    demo.stop()
                    break

                # 2. Check mouse input
                if mouse_fd:
                    try:
                        while True:
                            event_data = mouse_fd.read(EVENT_SIZE)
                            if not event_data: break
                            
                            _, _, ev_type, ev_code, ev_value = struct.unpack(EVENT_FORMAT, event_data)
                            if ev_type == EV_KEY and ev_code == BTN_LEFT:
                                if ev_value == 1: # Mouse down
                                    click_start_time = time.time()
                                elif ev_value == 0: # Mouse up
                                    click_start_time = None
                    except (BlockingIOError, TypeError):
                        pass

                # 3. Check if 1-second hold is met
                if click_start_time and (time.time() - click_start_time) > HOLD_THRESHOLD:
                    demo.stop()
                    break

                time.sleep(0.05)

            if mouse_fd:
                mouse_fd.close()
            # =========================================================

        except KeyboardInterrupt:
            # Ctrl+C pressed. Stop the demo and exit the main loop forever
            if demo:
                demo.stop()
            keep_cycling = False 

        finally:
            last_seen_data = {}
            class_colors = {}
            session_start_time = None
            
            # --- FIX 3: Only extract data if demo actually initialized ---
            if demo is not None:
                # We loop through the pipes to grab the dictionaries from post_process
                for pipe in demo.infer_pipes:
                    if hasattr(pipe, 'post_proc') and hasattr(pipe.post_proc, 'last_seen'):
                        last_seen_data.update(pipe.post_proc.last_seen)
                        class_colors.update(pipe.post_proc.class_colors)
                    if hasattr(pipe.post_proc, 'start_time') and session_start_time is None:
                        session_start_time = pipe.post_proc.start_time

                # Standard cleanup
                utils.disable_curses_reports()
            
            # --- Show the final static frame! ---
            if last_seen_data:
                show_final_summary(last_seen_data, class_colors, session_start_time)

            if demo is not None:
                del demo

            if keep_cycling:
                print("\n[INFO] Cycle complete. Restarting in 1 second...")
                time.sleep(1)
            else:
                print("\n[INFO] Application Shutdown Complete.")

if __name__ == "__main__":
    main(sys.argv)
