# modules/detection_manager.py
"""目标检测与分割管理器

流程:YOLO-World 开放词汇检测 → 仅对置信度最高的目标做 SAM 分割。
分割提示 = 检测框 + 「中心正点 + 边缘暗点负点」,用于把物体与相连阴影区分开
(负点采样策略见 sample_precise_points,设计说明见 README)。

性能要点:SAM 的 set_image/predict 每帧只对最优目标执行一次。
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from scipy.ndimage import binary_fill_holes
from ultralytics import YOLOWorld
from segment_anything import sam_model_registry, SamPredictor
import time
import torch


@dataclass
class Detection:
    """检测结果数据类"""
    bbox: np.ndarray
    confidence: float
    class_name: str
    class_id: int
    mask: Optional[np.ndarray] = None  # SAM 掩码(仅最优目标有)

    def __str__(self):
        return f"{self.class_name}: {self.confidence:.2f} at {self.bbox}"


class DetectionManager:
    """检测管理类"""

    def __init__(self, config):
        self.config = config
        self.model: Optional[YOLOWorld] = None
        self.sam_predictor: Optional[SamPredictor] = None
        self.classes: List[str] = []
        self.set_classes(config.default_classes)
        self.im = None

        self.device = 'cuda'
        self._initialize_sam()

    def set_classes(self, classes: List[str]):
        """设置检测类别(触发模型重新加载)"""
        self.classes = classes
        self._initialize_model()

    def _initialize_model(self):
        """初始化YOLO模型"""
        try:
            if self.model is not None:
                del self.model
                import gc
                gc.collect()

            self.model = YOLOWorld(self.config.model_path)
            self.model.set_classes(self.classes)
            print(f"✅ 模型加载完成，检测类别: {', '.join([c for c in self.classes if c])}")
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            raise

    def _initialize_sam(self):
        """初始化SAM分割模型(失败时降级为无分割,不影响检测)"""
        try:
            device = torch.device(self.device)
            sam_checkpoint = getattr(self.config, 'sam_checkpoint', 'sam_vit_h_4b8939.pth')
            model_type = getattr(self.config, 'sam_model_type', 'vit_h')

            print(f"🔄 正在加载SAM模型 ({model_type}): {sam_checkpoint}")

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            if self.device == "cuda":
                sam.float()  # 保持全精度;如需提速可改 sam.half() 并同步predictor输入

            self.sam_predictor = SamPredictor(sam)
            print("✅ SAM模型加载完成")

        except Exception as e:
            print(f"❌ SAM模型初始化失败(将跳过分割): {e}")
            self.sam_predictor = None

    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], List[Detection]]:
        """
        在图像上运行检测 + 最优目标分割

        Returns:
            (标注后的图像, 检测结果列表)
        """
        if self.model is None or frame is None:
            return None, []

        start_time = time.time()

        try:
            self.im = frame.copy()
            results = self.model.predict(
                self.im,
                conf=self.config.confidence_threshold,
                verbose=False,
                max_det=self.config.max_detections,
                retina_masks=True,
            )

            detections: List[Detection] = []
            annotated_frame = frame.copy()

            if results and results[0].boxes is not None:
                boxes = results[0].boxes

                # 第一步:找置信度最高的框
                best_box_idx = None
                max_conf = 0.0
                for i, box in enumerate(boxes):
                    current_conf = float(box.conf)
                    if current_conf > max_conf:
                        max_conf = current_conf
                        best_box_idx = i

                # 第二步:整图只做一次 set_image(这是 SAM 的主要开销之一)
                if best_box_idx is not None and self.sam_predictor is not None:
                    self.sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # 第三步:构建检测结果;SAM 分割只对最优目标执行一次
                for i, box in enumerate(boxes):
                    detection = self._process_box(box, results[0], i)

                    if i == best_box_idx and self.sam_predictor is not None:
                        bbox = box.xyxy.cpu().numpy().astype(int).flatten()
                        detection.mask = self._generate_sam_mask(bbox, frame)

                    detections.append(detection)
                    annotated_frame = self._draw_detection(annotated_frame, detection, i)

            if not detections:
                # 2026-10-16: ultralytics 8.3.x 与 v8.1 时代 yolov8s-world 权重 set_classes
                # 不兼容,文本嵌入失效恒返回空框;仿真彩块场景用 HSV 颜色检测兜底。
                fb = self._color_fallback(frame)
                if fb:
                    detections = fb
                    for i, d in enumerate(detections):
                        annotated_frame = self._draw_detection(annotated_frame, d, i)

            if self.config.display_stats:
                annotated_frame = self._add_stats(annotated_frame, detections, start_time)

            return annotated_frame, detections

        except Exception as e:
            print(f"❌ 检测失败: {e}")
            import traceback
            traceback.print_exc()
            return None, []

    def _color_fallback(self, frame: np.ndarray) -> List[Detection]:
        """HSV 颜色兜底检测(灰桌上找饱和色块),掩码直接取自颜色分割。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ranges = {
            'red': [(0, 100, 60, 10), (170, 100, 60, 179)],
            'green': [(40, 100, 60, 85)],
            'blue': [(100, 100, 60, 130)],
        }
        h, w = frame.shape[:2]
        out = []
        for color, spans in ranges.items():
            m = np.zeros((h, w), np.uint8)
            for hlo, slo, vlo, hhi in spans:
                m |= cv2.inRange(hsv, np.array([hlo, slo, vlo]), np.array([hhi, 255, 255]))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) < 400:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            out.append(Detection(
                bbox=np.array([x, y, x + bw, y + bh]),
                confidence=0.9,
                class_name=f'{color} cube',
                class_id=len(out),
                mask=mask,
            ))
        return out

    def sample_precise_points(self, frame: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        针对「阴影与物体连接」的提示点采样:
        1. 中心区域采正点,锚定物体本体(以中心亮度为参考基准);
        2. 框内边缘只采「明显更暗」的点作为负点,抑制阴影,避免误伤本体边缘。
        """
        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        points = []
        labels = []

        # 第一步:中心 30% 区域的平均亮度作为基准
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        core_x1, core_y1 = max(x1, cx - w // 7), max(y1, cy - h // 7)
        core_x2, core_y2 = min(x2, cx + w // 7), min(y2, cy + h // 7)
        core_brightness = np.mean(gray[core_y1:core_y2, core_x1:core_x2])
        # 阴影判定阈值:比中心暗 30% 以上
        shadow_brightness_thresh = core_brightness * 0.7

        # 第二步:中心正点
        for _ in range(4):
            px = np.random.randint(core_x1, core_x2)
            py = np.random.randint(core_y1, core_y2)
            points.append([px, py])
            labels.append(1)

        # 第三步:四条边缘各采最多 3 个暗点作负点(有尝试上限,避免卡死)
        edge_margin = min(w, h) // 5
        edge_regions = [
            (x1, y1, x2, y1 + edge_margin),  # 上
            (x1, y2 - edge_margin, x2, y2),  # 下
            (x1, y1, x1 + edge_margin, y2),  # 左
            (x2 - edge_margin, y1, x2, y2),  # 右
        ]

        for (ex1, ey1, ex2, ey2) in edge_regions:
            sampled = 0
            attempts = 0
            max_attempts = 10
            while sampled < 3 and attempts < max_attempts:
                px = np.random.randint(ex1, ex2)
                py = np.random.randint(ey1, ey2)
                if gray[py, px] < shadow_brightness_thresh:
                    points.append([px, py])
                    labels.append(0)
                    sampled += 1
                attempts += 1

        return np.array(points), np.array(labels)

    def _process_box(self, box, result, index: int) -> Detection:
        """处理单个检测框(不做SAM,掩码由 detect() 仅对最优目标生成)"""
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        if 0 <= class_id < len(self.classes):
            class_name = self.classes[class_id] if self.classes[class_id] else "unknown"
        else:
            class_name = f"class_{class_id}"

        return Detection(
            bbox=bbox,
            confidence=confidence,
            class_name=class_name,
            class_id=class_id,
            mask=None,
        )

    def _generate_sam_mask(self, bbox: np.ndarray, frame: np.ndarray) -> Optional[np.ndarray]:
        """框 + 精准点提示的SAM分割,含轻量后处理"""
        try:
            # 1. 采样「中心正点+边缘暗点负点」
            point_coords, point_labels = self.sample_precise_points(frame, bbox)

            # 2. SAM 预测
            input_box = np.array([bbox])
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=input_box,
                multimask_output=False,
            )

            if masks is None or len(masks) == 0:
                return None

            # 3. 后处理:填洞 + 只保留最大连通域(去阴影残片)
            best_mask = masks[0]
            mask_binary = (best_mask > 0.5).astype(np.uint8) * 255
            mask_binary = binary_fill_holes(mask_binary).astype(np.uint8) * 255

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
            if num_labels > 2:
                largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                mask_binary[labels != largest_label] = 0

            return mask_binary

        except Exception as e:
            print(f"⚠️ SAM掩码生成失败: {e}")
            return None

    def _draw_detection(self, image: np.ndarray, detection: Detection, index: int) -> np.ndarray:
        """绘制单个检测结果"""
        colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        color = colors[index % len(colors)]

        x1, y1, x2, y2 = detection.bbox

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{detection.class_name} {detection.confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(image,
                      (x1, y1 - label_height - 5),
                      (x1 + label_width, y1),
                      color, -1)
        cv2.putText(image, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image

    def _add_stats(self, image: np.ndarray, detections: List[Detection], start_time: float) -> np.ndarray:
        """添加统计信息到图像"""
        elapsed_ms = (time.time() - start_time) * 1000
        stats_y = 30

        cv2.putText(image, f"Classes: {len([c for c in self.classes if c])}",
                    (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Detections: {len(detections)}",
                    (10, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Time: {elapsed_ms:.0f}ms",
                    (10, stats_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        class_text = f"Classes: {', '.join([c for c in self.classes if c])}"
        cv2.putText(image, class_text[:50], (10, stats_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image
