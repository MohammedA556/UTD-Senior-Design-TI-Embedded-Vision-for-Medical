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
import time

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def draw_rounded_rectangle(frame, pt1, pt2, color, thickness=-1, radius=8):
    """
    Draw a rectangle with rounded corners.
    
    Args:
        frame: Image to draw on
        pt1: Top-left corner (x, y)
        pt2: Bottom-right corner (x, y)
        color: Color in BGR format
        thickness: Line thickness (-1 for filled)
        radius: Corner radius in pixels
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    if thickness == -1:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    
    return frame


def draw_confidence_bar(frame, x, y, confidence, bar_width=100, bar_height=8):
    """
    Draw a color-coded confidence bar with percentage.
    
    Args:
        frame: Image to draw on
        x, y: Top-left position of the bar
        confidence: Confidence value (0.0 to 1.0)
        bar_width: Width of the bar in pixels
        bar_height: Height of the bar in pixels
    """
    draw_rounded_rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1, 4)
    
    fill_width = int(bar_width * confidence)
    if confidence >= 0.7:
        bar_color = (136, 255, 0)
    elif confidence >= 0.5:
        bar_color = (0, 215, 255)
    else:
        bar_color = (107, 107, 255)
    
    if fill_width > 0:
        draw_rounded_rectangle(frame, (x, y), (x + fill_width, y + bar_height), bar_color, -1, 4)
    
    cv2.putText(frame, f"{int(confidence * 100)}%", (x + bar_width + 8, y + 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    return frame


def create_title_frame(title, width, height):
    frame = np.zeros((height, width, 3), np.uint8)
    if title != None:
        frame = cv2.putText(
            frame,
            "UT Dallas Senior Design Medvision Bots - Texas Instruments Edge",
            (40, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 212, 0),
            2,
        )
        frame = cv2.putText(
            frame, title, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (136, 255, 0), 2
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
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()
        self.inference_time = 0
        if flow.debug_config and flow.debug_config.post_proc:
            self.debug = debug.Debug(flow.debug_config, "post")
    
    def update_performance_metrics(self, inference_time_ms=0):
        """
        Update FPS and performance metrics.
        """
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time
        
        self.inference_time = inference_time_ms
    
    def draw_performance_panel(self, frame, detection_count=0):
        """
        Draw performance metrics panel in top-left corner.
        """
        panel_width = 280
        panel_height = 110
        
        draw_rounded_rectangle(frame, (10, 10), (10 + panel_width, 10 + panel_height), 
                             (40, 20, 0), -1, 8)
        draw_rounded_rectangle(frame, (10, 10), (10 + panel_width, 10 + panel_height), 
                             (255, 212, 0), 2, 8)
        
        cv2.putText(frame, "Performance Metrics", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 212, 0), 1)
        
        cv2.putText(frame, "FPS:", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        fps_color = (136, 255, 0) if self.fps >= 25 else (0, 215, 255) if self.fps >= 15 else (68, 68, 255)
        cv2.putText(frame, f"{self.fps:.1f}", (120, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fps_color, 1)
        
        cv2.putText(frame, "Inference Time:", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"{self.inference_time:.0f} ms", (120, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 212, 0), 1)
        
        cv2.putText(frame, "Detections:", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, str(detection_count), (120, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (136, 255, 0), 1)
        
        cv2.circle(frame, (270, 25), 5, (68, 68, 255), -1)
        
        return frame

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
        start_time = time.time()
        results = np.squeeze(results)
        img = self.overlay_topN_classnames(img, results)
        self.inference_time = (time.time() - start_time) * 1000

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
        self.update_performance_metrics()
        frame = self.draw_performance_panel(frame, detection_count=self.model.topN)
        
        orig_width = frame.shape[1]
        orig_height = frame.shape[0]
        row_size = 45 * orig_width // 1280
        font_size = orig_width / 1280 * 0.6
        N = self.model.topN
        topN_classes = np.argsort(results)[: (-1 * N) - 1 : -1]
        title_text = "Recognized Classes (Top %d)" % N
        font = cv2.FONT_HERSHEY_SIMPLEX

        start_y = 140
        draw_rounded_rectangle(frame, (0, start_y), (500, start_y + 40), (80, 40, 0), -1, 8)
        cv2.putText(frame, title_text, (10, start_y + 25), font, 0.7, (136, 255, 0), 2)
        
        y = start_y + 70
        rank = 1
        for idx in topN_classes:
            confidence = results[idx]
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

            draw_rounded_rectangle(frame, (5, y - 25), (495, y + 10), (80, 40, 0), -1, 6)
            
            cv2.putText(frame, f"{rank}.", (15, y), font, 0.6, (255, 212, 0), 2)
            cv2.putText(frame, class_name, (45, y), font, 0.5, (255, 255, 255), 1)
            
            draw_confidence_bar(frame, 280, y - 12, confidence, bar_width=100, bar_height=8)
            
            y += 45
            rank += 1
            if self.debug:
                self.debug_str += class_name + "\n"

        return frame


class PostProcessDetection(PostProcess):
    def __init__(self, flow):
        super().__init__(flow)

    def __call__(self, img, results):
        """
        Post process function for detection
        Args:
            img: Input frame
            results: output of inference
        """
        start_time = time.time()
        
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

        detection_count = 0
        for b in bbox:
            if b[5] > self.model.viz_threshold:
                detection_count += 1
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

                img = self.overlay_bounding_box(img, b, class_name, color)
        
        self.inference_time = (time.time() - start_time) * 1000
        self.update_performance_metrics()
        img = self.draw_performance_panel(img, detection_count)

        if self.debug:
            self.debug.log(self.debug_str)
            self.debug_str = ""

        return img

    def overlay_bounding_box(self, frame, box, class_name, color):
        """
        draw bounding box at given co-ordinates.

        Args:
            frame (numpy array): Input image where the overlay should be drawn
            bbox : Bounding box co-ordinates in format [X1 Y1 X2 Y2]
            class_name : Name of the class to overlay
        """
        confidence = box[5]
        box_coords = [
            int(box[0] * frame.shape[1]),
            int(box[1] * frame.shape[0]),
            int(box[2] * frame.shape[1]),
            int(box[3] * frame.shape[0]),
        ]

        box_color = color
        
        draw_rounded_rectangle(frame, (box_coords[0], box_coords[1]), 
                             (box_coords[2], box_coords[3]), box_color, 3, 8)
        
        label_width = 180
        label_height = 50
        label_x = box_coords[0] + 10
        label_y = box_coords[1] - label_height - 5
        
        if label_y < 0:
            label_y = box_coords[1] + 5
        
        draw_rounded_rectangle(frame, (label_x, label_y), 
                             (label_x + label_width, label_y + label_height), 
                             box_color, -1, 6)
        
        cv2.putText(frame, class_name, (label_x + 8, label_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(frame, "Confidence:", (label_x + 8, label_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        draw_confidence_bar(frame, label_x + 75, label_y + 28, confidence, 
                          bar_width=90, bar_height=6)

        if self.debug:
            self.debug_str += class_name
            self.debug_str += str(box_coords) + "\n"

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
        start_time = time.time()
        output = np.squeeze(results[0])

        scale_x = img.shape[1] / self.model.resize[0]
        scale_y = img.shape[0] / self.model.resize[1]

        det_bboxes, det_scores, det_labels, kpts = (
            np.array(output[:, 0:4]),
            np.array(output[:, 4]),
            np.array(output[:, 5]),
            np.array(output[:, 6:]),
        )
        
        detection_count = 0
        for idx in range(len(det_bboxes)):
            det_bbox = det_bboxes[idx]
            kpt = kpts[idx]
            if det_scores[idx] > self.model.viz_threshold:
                detection_count += 1
                det_bbox[..., (0, 2)] *= scale_x
                det_bbox[..., (1, 3)] *= scale_y

                dataset_idx = int(det_labels[idx])
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

                draw_rounded_rectangle(img, 
                                     (int(det_bbox[0]), int(det_bbox[1])),
                                     (int(det_bbox[2]), int(det_bbox[3])),
                                     (136, 255, 0), 3, 12)

                label_width = 150
                label_height = 30
                label_x = int(det_bbox[0]) + 10
                label_y = int(det_bbox[1]) - label_height - 5
                
                if label_y < 0:
                    label_y = int(det_bbox[1]) + 5
                
                draw_rounded_rectangle(img, (label_x, label_y),
                                     (label_x + label_width, label_y + label_height),
                                     (136, 255, 0), -1, 6)
                
                cv2.putText(img, class_name, (label_x + 8, label_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                confidence_text = f"{int(det_scores[idx] * 100)}%"
                cv2.putText(img, confidence_text, (label_x + label_width - 35, label_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                num_kpts = len(kpt) // 3
                keypoint_positions = []
                for kidx in range(num_kpts):
                    kx, ky, conf = kpt[3 * kidx], kpt[3 * kidx + 1], kpt[3 * kidx + 2]
                    kx = int(kx * scale_x)
                    ky = int(ky * scale_y)
                    keypoint_positions.append((kx, ky, conf))

                for sk in skeleton:
                    if sk[0] - 1 < len(keypoint_positions) and sk[1] - 1 < len(keypoint_positions):
                        pos1 = keypoint_positions[sk[0] - 1]
                        pos2 = keypoint_positions[sk[1] - 1]
                        
                        if pos1[2] > 0.5 and pos2[2] > 0.5:
                            cv2.line(img, (pos1[0], pos1[1]), (pos2[0], pos2[1]), 
                                   (255, 157, 0), 3)

                for kx, ky, conf in keypoint_positions:
                    if conf > 0.5:
                        cv2.circle(img, (kx, ky), 6, (255, 157, 0), -1)
                        cv2.circle(img, (kx, ky), 3, (255, 255, 255), -1)

        self.inference_time = (time.time() - start_time) * 1000
        self.update_performance_metrics()
        img = self.draw_performance_panel(img, detection_count)

        return img