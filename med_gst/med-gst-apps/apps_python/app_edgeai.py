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
    """
    Show summary after session end, displaying the last occurence of each class
    Where it terminates after 60 seconds or if the user presses ENTER
    """
    if not last_seen:
        return 

    # Standard canvas 
    height, width = 1080, 1920 
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30) 
    # Title
    cv2.putText(canvas, "SESSION SUMMARY: FINAL DETECTIONS", (40, 80), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    # Timeline UI element
    max_time = max(last_seen.values()) if last_seen else 1.0
    timeline_y, margin_x, timeline_w = 180, 80, width - 160
    cv2.line(canvas, (margin_x, timeline_y), (margin_x + timeline_w, timeline_y), (150, 150, 150), 2)
    # In order of chronology, display the class and count pair at timeline point
    sorted_items = sorted(last_seen.items(), key=lambda x: x[1], reverse=True)
    for i, (cls_tuple, t) in enumerate(sorted_items[:10]): # Show top 10
        cls, count = cls_tuple
        color = class_colors.get(cls, (0, 255, 0))
        # Point and Line Element
        y = 300 + (i * 45)
        line_y = timeline_y + 20 + (i*5)
        x_pos = margin_x + int((t / max_time) * timeline_w)
        abs_timestamp = start_time + t 
        time_str = datetime.fromtimestamp(abs_timestamp).strftime('%H:%M:%S')
        cv2.circle(canvas, (x_pos, timeline_y), 10, color, -1)
        cv2.line(canvas, (x_pos, timeline_y), (x_pos, line_y), color, 1) 
        cv2.putText(canvas, f"{i}", (x_pos, line_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Provide class last detection information such as the exact timestamp
        text = f"{i}. {cls}, C={count}: Last seen at t={t:.2f}, Timestamp: {time_str}"
        cv2.putText(canvas, text, (margin_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Save a temporary image that holds the final display
    temp_path = "/tmp/final_summary.jpg"
    cv2.imwrite(temp_path, canvas)

    # Launch GStreamer to display the image on the hardware sink
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

    # Display timeline until timer runs out or user presses ENTER in terminal
    try:
        # Yield until user presses enter or until 60 seconds have passed
        i, o, e = select.select([sys.stdin], [], [], 60)
        if i:
            sys.stdin.readline() # Clear the keypress from buffer
            print("Terminating via keypress...")
    except KeyboardInterrupt:
        # If user uses CTRL+C to end program
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

    # Event format for the mouse interrupt in the AM62A Linux SDK
    EVENT_FORMAT = "llHHi" 
    EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
    EV_KEY = 1
    BTN_LEFT = 272

    keep_cycling = True
    # Keep starting new sessions after the last session ends until program is terminated
    while keep_cycling:
        demo = None 
        try:
            # Reset config_parser state for each session
            importlib.reload(config_parser)
            utils.report_list.clear()

            # Load the config fresh 
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
            
            # Start the demo session instance
            print("\n[INFO] Starting/Restarting Demo Cycle...")
            demo = EdgeAIDemo(config)
            demo.start()

            if args.verbose:
                utils.print_stdout = True

            if not args.no_curses:
                utils.enable_curses_reports(demo.title)

            # Replacement for the demo's normal wait for end
            # Try to extract mouse data first
            try:
                mouse_fd = open("/dev/input/event0", "rb")
                os.set_blocking(mouse_fd.fileno(), False)
            except Exception as e:
                mouse_fd = None

            click_start_time = None
            # 1 Second hold to end cycle
            HOLD_THRESHOLD = 1.0 
            # Continue session until an event terminates session
            while True:
                # Terminate if demo ended internally 
                if all(i.stop_thread for i in demo.infer_pipes):
                    demo.stop()
                    break

                # Extract mouse holding duration
                if mouse_fd:
                    try:
                        while True:
                            event_data = mouse_fd.read(EVENT_SIZE)
                            if not event_data: break
                            # If mouse is down, set the new start time when the press started
                            _, _, ev_type, ev_code, ev_value = struct.unpack(EVENT_FORMAT, event_data)
                            if ev_type == EV_KEY and ev_code == BTN_LEFT:
                                if ev_value == 1: 
                                    click_start_time = time.time()
                                elif ev_value == 0: 
                                    click_start_time = None
                    except (BlockingIOError, TypeError):
                        pass

                # Terminate if 1-second hold is met
                if click_start_time and (time.time() - click_start_time) > HOLD_THRESHOLD:
                    demo.stop()
                    break
                
                time.sleep(0.05)
            # Close mouse event stream 
            if mouse_fd:
                mouse_fd.close()
            # =========================================================

        except KeyboardInterrupt:
            # Stop cycling sessions and terminate program on CTRL+C
            if demo:
                demo.stop()
            keep_cycling = False 

        finally:
            last_seen_data = {}
            class_colors = {}
            session_start_time = None
            
            # If there was a last session then:
            if demo is not None:
                # Loop through the inference pipes to get relevant data 
                # such as last seen of each class, count and start time
                for pipe in demo.infer_pipes:
                    if hasattr(pipe, 'post_proc') and hasattr(pipe.post_proc, 'last_seen'):
                        last_seen_data.update(pipe.post_proc.last_seen)
                        class_colors.update(pipe.post_proc.class_colors)
                    if hasattr(pipe.post_proc, 'start_time') and session_start_time is None:
                        session_start_time = pipe.post_proc.start_time

                utils.disable_curses_reports()
            
            # If there is last seen data then show the final summary timeline screen
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
