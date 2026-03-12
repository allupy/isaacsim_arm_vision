# modules/detection_manager.py
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from ultralytics import YOLOWorld
from segment_anything import sam_model_registry,SamPredictor
import time
import torch

@dataclass
class Detection:
    """
    检测结果数据类
    使用yolo作为目标检测
    后续添加位置姿态估计
    """
    bbox: np.ndarray
    confidence: float
    class_name: str
    class_id: int
    mask: Optional[np.ndarray] = None  # 新增：掩码
    
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
        # 初始化SAM
        self._initialize_sam()
    
    def set_classes(self, classes: List[str]):
        """设置检测类别"""
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
        """初始化SAM分割模型"""
        try:

            device = torch.device(self.device)
            # SAM配置
            sam_checkpoint = getattr(self.config, 'sam_checkpoint', 'sam_vit_h_4b8939.pth')
            model_type = getattr(self.config, 'sam_model_type', 'vit_h')
            
            print(f"🔄 正在加载SAM模型 ({model_type})...")
            
            # 加载SAM模型
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            
            # 如果使用CUDA，启用半精度
            if self.device == "cuda":
                sam.float()
            
            self.sam_predictor = SamPredictor(sam)
            print(f"✅ SAM模型加载完成")
        
        except Exception as e:  # 这里需要添加except
            print(f"❌ SAM模型初始化失败: {e}")
            self.sam_predictor = None  # 设置None而不是重新抛出异常


    # def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], List[Detection]]:
    #     """
    #     在图像上运行检测
        
    #     Args:
    #         frame: 输入图像 (BGR格式)
            
    #     Returns:
    #         tuple: (标注后的图像, 检测结果列表)
    #     """
    #     if self.model is None or frame is None:
    #         return None, []
        
    #     start_time = time.time()
        
    #     try:
    #         self.im = frame.copy()


    #         # 运行检测
    #         results = self.model.predict(
    #             self.im ,
    #             conf=self.config.confidence_threshold,
    #             verbose=False,
    #             max_det=self.config.max_detections,
    #             retina_masks=True,  # 使用高质量掩码
    #         )
            
    #         detections = []
    #         annotated_frame = frame.copy()
            
    #         if results and results[0].boxes is not None:
    #             # h, w = frame.shape[:2]
    #             # # 处理每个检测结果
    #             # self.sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #特别耗时
    #             # result[0]是什么?

    #             for i, box in enumerate(results[0].boxes):
    #                 detection = self._process_box_with_mask(box, results[0], i)
    #                 detections.append(detection)
                    
    #                 # 绘制到图像上
    #                 annotated_frame = self._draw_detection(annotated_frame, detection, i)
            
    #         # 添加统计信息
    #         if self.config.display_stats:
    #             annotated_frame = self._add_stats(annotated_frame, detections, start_time)
            
    #         return annotated_frame, detections
            
    #     except Exception as e:
    #         print(f"❌ 检测失败: {e}")
    #         return None, []

    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], List[Detection]]:
        """
        优化版：保留原始安全逻辑 + 只对高置信度框做SAM分割
        """
        if self.model is None or frame is None:
            return None, []
        
        start_time = time.time()
        
        try:
            self.im = frame.copy()

            # 运行检测
            results = self.model.predict(
                self.im,
                conf=self.config.confidence_threshold,
                verbose=False,
                max_det=self.config.max_detections,
                retina_masks=True,
            )
            
            detections = []
            annotated_frame = frame.copy()
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                best_box_idx = None  # 最优框的索引
                max_conf = 0.0       # 最高置信度
                
                # 第一步：先找到置信度最高的框的索引（不手动转数组）
                for i, box in enumerate(boxes):
                    current_conf = float(box.conf)
                    if current_conf > max_conf:
                        max_conf = current_conf
                        best_box_idx = i
                
                # 第二步：只对最优框初始化SAM（节约时间）
                if best_box_idx is not None and self.sam_predictor is not None:
                    self.sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    print("有找到目标")
                
                # 第三步：遍历所有框（复用原始安全逻辑）
                for i, box in enumerate(boxes):
                    # 生成detection（原始逻辑，无数组维度问题）
                    detection = self._process_box_with_mask(box, results[0], i)
                    
                    # 仅对最优框调用SAM分割
                    if i == best_box_idx and self.sam_predictor is not None:
                        print("准备分割掩码")
                        # 从box对象提取合规的bbox（交给_generate_sam_mask处理）
                        bbox = box.xyxy.cpu().numpy().astype(int).flatten()
                        detection.mask = self._generate_sam_mask(bbox, frame)
                    
                    detections.append(detection)
                    annotated_frame = self._draw_detection(annotated_frame, detection, i)
                    print("输出结果")
            
            # 添加统计信息
            if self.config.display_stats:
                annotated_frame = self._add_stats(annotated_frame, detections, start_time)
            
            return annotated_frame, detections
            
        except Exception as e:
            print(f"❌ 检测失败: {e}")
            import traceback
            traceback.print_exc()
            return None, []


    def sample_precise_points(self, frame: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        针对「阴影与物体连接」的优化采样：
        1. 中心正点锚定本体
        2. 边缘只采「暗点」当负点（避开物体本体边缘）
        修复：增加循环退出上限，避免卡死
        """
        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        points = []
        labels = []

        # --- 第一步：计算物体中心亮度（作为参考基准）---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 取中心30%区域的平均亮度
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        core_x1, core_y1 = max(x1, cx - w//7), max(y1, cy - h//7)
        core_x2, core_y2 = min(x2, cx + w//7), min(y2, cy + h//7)
        core_brightness = np.mean(gray[core_y1:core_y2, core_x1:core_x2])
        # 阴影判定阈值：比中心暗30%以上（可微调）
        shadow_brightness_thresh = core_brightness * 0.7

        # --- 第二步：采样中心正点（锚定本体）---
        for _ in range(4):
            px = np.random.randint(core_x1, core_x2)
            py = np.random.randint(core_y1, core_y2)
            points.append([px, py])
            labels.append(1)  # 正点：物体本体

        # --- 第三步：采样边缘「暗点」当负点（只抑制阴影）---
        edge_margin = min(w, h) // 5
        edge_regions = [
            (x1, y1, x2, y1 + edge_margin),  # 上边缘
            (x1, y2 - edge_margin, x2, y2),  # 下边缘
            (x1, y1, x1 + edge_margin, y2),  # 左边缘
            (x2 - edge_margin, y1, x2, y2)   # 右边缘
        ]

        for (ex1, ey1, ex2, ey2) in edge_regions:
            sampled = 0
            max_attempts = 10  # 修复：设置最大尝试次数，避免无限循环
            attempts = 0
            
            # 最多尝试10次，找到足够的暗点（避免采到物体本体）
            while sampled < 3 and attempts < max_attempts:
                px = np.random.randint(ex1, ex2)
                py = np.random.randint(ey1, ey2)
                # 只采「比中心暗30%以上」的点当负点
                if gray[py, px] < shadow_brightness_thresh:
                    points.append([px, py])
                    labels.append(0)  # 负点：阴影/背景
                    sampled += 1
                attempts += 1  # 修复：每次循环计数，达到上限退出

        return np.array(points), np.array(labels)

    def _process_box_with_mask(self, box, result, index: int) -> Detection:
        """
        处理单个检测框，包含掩码提取
        """
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        
        # 获取类别名称
        if 0 <= class_id < len(self.classes):
            class_name = self.classes[class_id] if self.classes[class_id] else "unknown"
        else:
            class_name = f"class_{class_id}"
        
        # 初始化掩码为None
        mask = self._generate_sam_mask(bbox,self.im)
        #mask = self._generate_sam_mask(bbox)


           
        # 修复：添加缺少的逗号
        return Detection(
            bbox=bbox,
            confidence=confidence,
            class_name=class_name,
            class_id=class_id,  # ← 这里加逗号
            mask=mask
        )
    
    def _generate_sam_mask(self, bbox: np.ndarray, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            # 1. 采样「中心正点+边缘暗点负点」（适配连接阴影）
            point_coords, point_labels = self.sample_precise_points(frame, bbox)
            
            # 2. SAM 预测（框 + 精准点提示）
            input_box = np.array([bbox])
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=input_box,
                multimask_output=False,
            )
            
            if masks is None or len(masks) == 0:
                return None
            
            # 3. 轻量后处理（清理连接阴影的小残留）
            best_mask = masks[0]
            mask_binary = (best_mask > 0.5).astype(np.uint8) * 255
            
            # 填充物体本体的空洞（避免SAM切出小洞）
            from scipy.ndimage import binary_fill_holes
            mask_binary = binary_fill_holes(mask_binary).astype(np.uint8) * 255
            
            # 保留最大连通域（去掉阴影的小碎片）
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
            if num_labels > 2:
                largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                mask_binary[labels != largest_label] = 0
            
            return mask_binary
        
        except Exception as e:
            print(f"⚠️ SAM掩码生成失败: {e}")
            return None
    # def _generate_sam_mask(self, bbox: np.ndarray) -> Optional[np.ndarray]:
    #     """
    #     使用SAM为边界框生成分割掩码
    #     """
    #     try:
    #         # 准备输入框 (格式: [x_min, y_min, x_max, y_max])
    #         input_box = np.array([bbox])
            
    #         # 运行SAM预测
    #         masks, scores, _ = self.sam_predictor.predict(
    #             point_coords=None,
    #             point_labels=None,
    #             box=input_box,
    #             multimask_output=False,  # 只输出最佳掩码
    #         )
            
    #         if masks is None or len(masks) == 0:
    #             return None
            
    #         # 获取最佳掩码并二值化
    #         best_mask = masks[0]
    #         mask_binary = (best_mask > 0.5).astype(np.uint8) * 255
            
    #         return mask_binary
            
    #     except Exception as e:
    #         print(f"⚠️ SAM掩码生成失败: {e}")
    #         return None


    def _draw_detection(self, image: np.ndarray, detection: Detection, index: int) -> np.ndarray:
        """绘制单个检测结果"""
        colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        color = colors[index % len(colors)]
        
        x1, y1, x2, y2 = detection.bbox
        thickness = 2
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 绘制标签
        label = f"{detection.class_name} {detection.confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # 标签背景
        cv2.rectangle(image,
                      (x1, y1 - label_height - 5),
                      (x1 + label_width, y1),
                      color, -1)
        
        # 标签文本
        cv2.putText(image, label,
                   (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return image
    
    def _add_stats(self, image: np.ndarray, detections: List[Detection], start_time: float) -> np.ndarray:
        """添加统计信息到图像"""
        elapsed_ms = (time.time() - start_time) * 1000
        stats_y = 30
        
        # 统计信息
        cv2.putText(image, f"Classes: {len([c for c in self.classes if c])}",
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Detections: {len(detections)}",
                   (10, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Time: {elapsed_ms:.0f}ms",
                   (10, stats_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示类别
        class_text = f"Classes: {', '.join([c for c in self.classes if c])}"
        cv2.putText(image, class_text[:50], (10, stats_y + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image