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

import cv2
import numpy as np
import copy
import debug
import utils
from collections import Counter
import time
import os
import csv
import math
from datetime import datetime

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def create_title_frame(title, width, height,
                       bottom_text="TEST", bottom_height=200,
                       bottom_bg="#3D2727", bottom_text_color="#FFFFFF"):
    frame = np.zeros((height, width, 3), np.uint8)
    if title != None:
        frame = cv2.putText(
            frame,
            "UT Dallas Senior Design Medvision Bots - Texas Instruments Edge",
            (40, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            2,
        )
        frame = cv2.putText(
            frame, title, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )


    return frame


def overlay_model_name(frame, model_name, start_x, start_y, width, height):
    row_size = 40 * width // 1280
    font_size = width / 1280
    cv2.putText(
        frame,
        "Model : " + model_name,
        (start_x + 5, start_y - row_size // 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        2,
    )
    return frame


class PostProcess:
    """
    Class to create a post process context
    """

    def __init__(self, flow):
        self.flow = flow
        self.model = flow.model
        self.debug = None
        self.debug_str = ""
        if flow.debug_config and flow.debug_config.post_proc:
            self.debug = debug.Debug(flow.debug_config, "post")

    def get(flow):
        """
        Create a object of a subclass based on the task type
        """
        if flow.model.task_type == "classification":
            return PostProcessClassification(flow)
        elif flow.model.task_type == "detection":
            return PostProcessDetection(flow)
        elif flow.model.task_type == "segmentation":
            return PostProcessSegmentation(flow)
        elif flow.model.task_type == "keypoint_detection":
            return PostProcessKeypointDetection(flow)


class PostProcessClassification(PostProcess):
    def __init__(self, flow):
        super().__init__(flow)

    def __call__(self, img, results):
        """
        Post process function for classification
        Args:
            img: Input frame
            results: output of inference
        """
        results = np.squeeze(results)
        img = self.overlay_topN_classnames(img, results)

        if self.debug:
            self.debug.log(self.debug_str)
            self.debug_str = ""

        return img

    def overlay_topN_classnames(self, frame, results):
        """
        Process the results of the image classification model and draw text
        describing top 5 detected objects on the image.

        Args:
            frame (numpy array): Input image in BGR format where the overlay should
        be drawn
            results (numpy array): Output of the model run
        """
        orig_width = frame.shape[1]
        orig_height = frame.shape[0]
        row_size = 40 * orig_width // 1280
        font_size = orig_width / 1280
        N = self.model.topN
        topN_classes = np.argsort(results)[: (-1 * N) - 1 : -1]
        title_text = "Recognized Classes (Top %d):" % N
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_size, _ = cv2.getTextSize(title_text, font, font_size, 2)

        bg_top_left = (0, (2 * row_size) - text_size[1] - 5)
        bg_bottom_right = (text_size[0] + 10, (2 * row_size) + 3 + 5)
        font_coord = (5, 2 * row_size)

        cv2.rectangle(frame, bg_top_left, bg_bottom_right, (5, 11, 120), -1)

        cv2.putText(
            frame,
            title_text,
            font_coord,
            font,
            font_size,
            (0, 255, 0),
            2,
        )
        row = 3
        for idx in topN_classes:
            idx = idx + self.model.label_offset
            if idx in self.model.dataset_info:
                class_name = self.model.dataset_info[idx].name
                if not class_name:
                    class_name = "UNDEFINED"
                if self.model.dataset_info[idx].supercategory:
                    class_name = (
                        self.model.dataset_info[idx].supercategory + "/" + class_name
                    )
            else:
                class_name = "UNDEFINED"

            text_size, _ = cv2.getTextSize(class_name, font, font_size, 2)

            bg_top_left = (0, (row_size * row) - text_size[1] - 5)
            bg_bottom_right = (text_size[0] + 10, (row_size * row) + 3 + 5)
            font_coord = (5, row_size * row)

            cv2.rectangle(frame, bg_top_left, bg_bottom_right, (5, 11, 120), -1)
            cv2.putText(
                frame,
                class_name,
                font_coord,
                font,
                font_size,
                (255, 255, 0),
                2,
            )
            row = row + 1
            if self.debug:
                self.debug_str += class_name + "\n"

        return frame


class PostProcessDetection(PostProcess):
    def __init__(self, flow):
        super().__init__(flow)
        self.start_time = None
        self.history = []          # Will store tuples of: (timestamp, frame_counts_dict)
        self.class_colors = {}     # Maps class_name -> official bounding box color
        self.last_seen = {}
        self.accum_frame_counts = Counter()
        self.frames_since_last_avg = 0

        # --- NEW: CSV Logging State ---
        self.log_dir = "./medvision_logs"  # Configurable: Change this path to wherever you want the logs saved
        self.log_filepath = None
        self.last_logged_counts = None     # Tracks changes
        self.pending_log_entries = []      # Queue for the chunk dumper
        self.frames_since_last_dump = 0

    def _dump_logs_to_file(self):
        """Helper function to append pending logs to the CSV file safely."""
        if not self.pending_log_entries or not self.log_filepath:
            return
        
        try:
            # Open the file in 'append' mode ('a')
            with open(self.log_filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                for t, summary in self.pending_log_entries:
                    writer.writerow([f"{t:.2f}", summary])
            
            # Clear the queue after a successful write
            self.pending_log_entries.clear()
        except Exception as e:
            print(f"[WARNING] Could not write to log file: {e}")

    def __del__(self):
        """Attempt to dump any remaining logs when the program closes."""
        self._dump_logs_to_file()

    def __call__(self, img, results):
        """
        Post process function for detection
        Args:
            img: Input frame
            results: output of inference
        """
        # 1. Initialize start time and the CSV File
        if self.start_time is None:
            self.start_time = time.time()
            
            # Create the folder if it doesn't exist
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Generate file name: e.g., "detection_log_20240512_143000.csv"
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_filepath = os.path.join(self.log_dir, f"detection_log_{timestamp_str}.csv")
            
            # Write the CSV Header
            with open(self.log_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time (seconds)", "Detections"])
        
        current_time = time.time() - self.start_time

        orig_h, orig_w = img.shape[:2]
        # Increase UI height slightly (30%) to make room for the chart
        ui_height = int(orig_h * 0.30)
        video_height = orig_h - ui_height

        for i, r in enumerate(results):
            r = np.squeeze(r)
            if r.ndim == 1:
                r = np.expand_dims(r, 1)
            results[i] = r

        if self.model.shuffle_indices:
            results_reordered = []
            for i in self.model.shuffle_indices:
                results_reordered.append(results[i])
            results = results_reordered

        if results[-1].ndim < 2:
            results = results[:-1]

        bbox = np.concatenate(results, axis=-1)

        if self.model.formatter:
            if self.model.ignore_index == None:
                bbox_copy = copy.deepcopy(bbox)
            else:
                bbox_copy = copy.deepcopy(np.delete(bbox, self.model.ignore_index, 1))
            bbox[..., self.model.formatter["dst_indices"]] = bbox_copy[
                ..., self.model.formatter["src_indices"]
            ]

        if not self.model.normalized_detections:
            bbox[..., (0, 2)] /= self.model.resize[0]
            bbox[..., (1, 3)] /= self.model.resize[1]

        for b in bbox:
            if b[5] > self.model.viz_threshold:
                if type(self.model.label_offset) == dict:
                    class_name_idx = self.model.label_offset[int(b[4])]
                else:
                    class_name_idx = self.model.label_offset + int(b[4])

                if class_name_idx in self.model.dataset_info:
                    class_name = self.model.dataset_info[class_name_idx].name
                    if not class_name:
                        class_name = "UNDEFINED"
                    if self.model.dataset_info[class_name_idx].supercategory:
                        class_name = (
                            self.model.dataset_info[class_name_idx].supercategory
                            + "/"
                            + class_name
                        )
                    color = self.model.dataset_info[class_name_idx].rgb_color
                else:
                    class_name = "UNDEFINED"
                    color = (20, 220, 20)

                if class_name != "UNDEFINED":
                    self.accum_frame_counts[class_name] += 1
                    self.class_colors[class_name] = color

                img = self.overlay_bounding_box(img, b, class_name, color)

        if self.debug:
            self.debug.log(self.debug_str)
            self.debug_str = ""

        frame_counts = Counter()
        self.frames_since_last_avg += 1
        if self.frames_since_last_avg >= 3:
            for class_name, count in self.accum_frame_counts.items():
                avg_count = math.floor(0.5 + count / 3)
                if avg_count > 0:
                    frame_counts[class_name] += avg_count
                    self.last_seen[(class_name, avg_count)] = current_time
            self.accum_frame_counts.clear()
            self.frames_since_last_avg = 0
        elif self.history:
            frame_counts = self.history[-1][1]
        self.history.append((current_time, frame_counts))

        if self.last_logged_counts != frame_counts:
            # Format as a clean string for the CSV: "Scalpel: 1 | Suture: 2"
            summary_str = " | ".join([f"{k}: {v}" for k, v in frame_counts.items()]) if frame_counts else "NO DETECTIONS"
            
            # Add to the pending queue
            self.pending_log_entries.append((current_time, summary_str))
            
            # Update the last known state
            self.last_logged_counts = dict(frame_counts)

        # Trigger the file dump every 1000 frames
        self.frames_since_last_dump += 1
        if self.frames_since_last_dump >= 1000:
            self._dump_logs_to_file()
            self.frames_since_last_dump = 0

        # To prevent the array from growing infinitely and crashing RAM after hours of running,
        # we can optionally cap the history to the last 5000 frames (approx 2-3 mins at 30fps).
        # Remove this line if you want it to log infinitely forever.
        if len(self.history) > 5000:
            self.history.pop(0)

        # 2. Resize video and create canvas
        img_resized = cv2.resize(img, (orig_w, video_height))
        canvas = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        canvas[0:video_height, 0:orig_w] = img_resized

        # 3. Draw the UI Background
        ui_bg_color = (40, 30, 30) # Dark Charcoal
        cv2.rectangle(canvas, (0, video_height), (orig_w, orig_h), ui_bg_color, -1)
        cv2.line(canvas, (0, video_height), (orig_w, video_height), (0, 255, 0), 2)

        # 4. Text Summary Header
        summary_text = " | ".join([f"{n}: {c}" for n, c in frame_counts.items()]) if frame_counts else "NO DETECTIONS"
        cv2.putText(canvas, "LIVE ANALYTICS", (20, video_height + 25), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, summary_text, (250, video_height + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 5. --- DRAW THE DYNAMIC LINE CHART ---
        chart_x = 20
        chart_y = video_height + 40
        chart_w = orig_w - 40
        chart_h = ui_height - 60

        # Draw Chart Background Area (slightly darker)
        cv2.rectangle(canvas, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (50, 0, 0), -1)

        if len(self.history) > 1:
            # Determine scales
            max_time = max(current_time, 1.0)
            
            # Find the max object count across all history to scale the Y-axis
            max_count = 1
            all_seen_classes = set()
            for _, counts in self.history:
                if counts:
                    max_count = max(max_count, max(counts.values()))
                    all_seen_classes.update(counts.keys())
            
            # Draw grid lines for the chart
            cv2.line(canvas, (chart_x, chart_y + chart_h), (chart_x + chart_w, chart_y + chart_h), (100, 100, 100), 1)
            cv2.putText(canvas, f"Max: {max_count}", (chart_x + 5, chart_y + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            # Draw a line for each class observed so far
            for cls in all_seen_classes:
                pts = []
                color = self.class_colors.get(cls, (255, 255, 255))
                
                for t, counts in self.history:
                    count = counts.get(cls, 0)
                    
                    # Math to map (time, count) to (pixel_X, pixel_Y)
                    # X scales down automatically because t is divided by max_time
                    px = chart_x + int((t / max_time) * chart_w)
                    
                    # Y maps such that 0 is at the bottom (chart_y + chart_h) and max_count is at the top
                    py = chart_y + chart_h - int((count / max_count) * chart_h)
                    pts.append((px, py))
                
                # Draw the line efficiently
                pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts_arr], isClosed=False, color=color, thickness=2)

                # Draw a dot at the current end of the line
                last_pt = pts[-1]
                cv2.circle(canvas, last_pt, 4, color, -1)
        return canvas

    def overlay_bounding_box(self, frame, box, class_name, color):
        """
        draw bounding box at given co-ordinates.

        Args:
            frame (numpy array): Input image where the overlay should be drawn
            bbox : Bounding box co-ordinates in format [X1 Y1 X2 Y2]
            class_name : Name of the class to overlay
        """
        box = [
            int(box[0] * frame.shape[1]),
            int(box[1] * frame.shape[0]),
            int(box[2] * frame.shape[1]),
            int(box[3] * frame.shape[0]),
        ]

        box_color = color
        luma = ((66*(color[0])+129*(color[1])+25*(color[2])+128)>>8)+16
        if(luma >= 128):
            text_color = (0, 0, 0)
        else:
            text_color = (255, 255, 255)

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), box_color, 2)
        cv2.rectangle(
            frame,
            (int((box[2] + box[0]) / 2) - 5, int((box[3] + box[1]) / 2) + 5),
            (int((box[2] + box[0]) / 2) + 160, int((box[3] + box[1]) / 2) - 15),
            box_color,
            -1,
        )
        cv2.putText(
            frame,
            class_name,
            (int((box[2] + box[0]) / 2), int((box[3] + box[1]) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
        )

        if self.debug:
            self.debug_str += class_name
            self.debug_str += str(box) + "\n"

        return frame


class PostProcessSegmentation(PostProcess):
    def __call__(self, img, results):
        """
        Post process function for segmentation
        Args:
            img: Input frame
            results: output of inference
        """
        img = self.blend_segmentation_mask(img, results[0])

        return img

    def blend_segmentation_mask(self, frame, results):
        """
        Process the result of the semantic segmentation model and return
        an image color blended with the mask representing different color
        for each class

        Args:
            frame (numpy array): Input image in BGR format which should be blended
            results (numpy array): Results of the model run
        """

        mask = np.squeeze(results)

        if len(mask.shape) > 2:
            mask = mask[0]

        if self.debug:
            self.debug_str += str(mask.flatten()) + "\n"
            self.debug.log(self.debug_str)
            self.debug_str = ""

        # Resize the mask to the original image for blending
        org_image_rgb = frame
        org_width = frame.shape[1]
        org_height = frame.shape[0]

        mask_image_rgb = self.gen_segment_mask(mask)
        mask_image_rgb = cv2.resize(
            mask_image_rgb, (org_width, org_height), interpolation=cv2.INTER_LINEAR
        )

        blend_image = cv2.addWeighted(
            mask_image_rgb, 1 - self.model.alpha, org_image_rgb, self.model.alpha, 0
        )

        return blend_image

    def gen_segment_mask(self, inp):
        """
        Generate the segmentation mask from the result of semantic segmentation
        model. Creates an RGB image with different colors for each class.

        Args:
            inp (numpy array): Result of the model run
        """

        r_map = (inp * 10).astype(np.uint8)
        g_map = (inp * 20).astype(np.uint8)
        b_map = (inp * 30).astype(np.uint8)

        return cv2.merge((r_map, g_map, b_map))

class PostProcessKeypointDetection(PostProcess):

    def __init__(self, flow):
        super().__init__(flow)

    def __call__(self, img, results):
        """
        Post process function for keypoint detection
        Args:
            img: Input frame
            results: output of inference
        """
        output = np.squeeze(results[0])

        scale_x = img.shape[1] / self.model.resize[0]
        scale_y = img.shape[0] / self.model.resize[1]

        det_bboxes, det_scores, det_labels, kpts = (
            np.array(output[:, 0:4]),
            np.array(output[:, 4]),
            np.array(output[:, 5]),
            np.array(output[:, 6:]),
        )
        for idx in range(len(det_bboxes)):
            det_bbox = det_bboxes[idx]
            kpt = kpts[idx]
            if det_scores[idx] > self.model.viz_threshold:
                det_bbox[..., (0, 2)] *= scale_x
                det_bbox[..., (1, 3)] *= scale_y

                # Drawing bounding box
                img = cv2.rectangle(
                    img,
                    (int(det_bbox[0]), int(det_bbox[1])),
                    (int(det_bbox[2]), int(det_bbox[3])),
                    (0, 255, 0),
                    2,
                )

                dataset_idx = int(det_labels[idx])
                # Put Label
                if type(self.model.label_offset) == dict:
                    dataset_idx = self.model.label_offset[dataset_idx]
                else:
                    dataset_idx = self.model.label_offset + dataset_idx

                if dataset_idx in self.model.dataset_info:
                    class_name = self.model.dataset_info[dataset_idx].name
                    if not class_name:
                        class_name = "UNDEFINED"
                    if self.model.dataset_info[dataset_idx].supercategory:
                        class_name = (
                            self.model.dataset_info[dataset_idx].supercategory
                            + "/"
                            + class_name
                        )
                    skeleton = self.model.dataset_info[dataset_idx].skeleton
                    if not skeleton:
                        skeleton = []

                else:
                    class_name = "UNDEFINED"
                    skeleton = []

                cv2.putText(
                    img,
                    class_name,
                    (int(det_bbox[0]), int(det_bbox[1]) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

                # Drawing keypoints
                num_kpts = len(kpt) // 3
                for kidx in range(num_kpts):
                    kx, ky, conf = kpt[3 * kidx], kpt[3 * kidx + 1], kpt[3 * kidx + 2]
                    kx = int(kx * scale_x)
                    ky = int(ky * scale_y)
                    if conf > 0.5:
                        cv2.circle(img, (kx, ky), 3, (255, 0, 0), -1)

                # Drawing connections between keypoints
                for sk in skeleton:
                    pos1 = (kpt[(sk[0] - 1) * 3], kpt[(sk[0] - 1) * 3 + 1])
                    pos1 = (int(pos1[0] * scale_x), int(pos1[1] * scale_y))

                    pos2 = (kpt[(sk[1] - 1) * 3], kpt[(sk[1] - 1) * 3 + 1])
                    pos2 = (int(pos2[0] * scale_x), int(pos2[1] * scale_y))

                    conf1 = kpt[(sk[0] - 1) * 3 + 2]
                    conf2 = kpt[(sk[1] - 1) * 3 + 2]
                    if conf1 > 0.5 and conf2 > 0.5:
                        cv2.line(img, pos1, pos2, (255, 0, 0), 1)


        return img