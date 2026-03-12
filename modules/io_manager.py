# modules/io_manager.py
import os
import cv2
import json
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
from .detection_manager import Detection
import numpy as np 

class IOManager:
    """输入输出管理器"""
    
    def __init__(self, save_dir: str = "detection_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save_detection_result(self,
                             raw_image: np.ndarray,
                             result_image: np.ndarray,
                             detections: List[Detection],
                             classes: List[str],
                             prefix: str = "detection") -> Dict:
        """
        保存检测结果
        
        Returns:
            Dict: 保存的文件路径信息
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{timestamp}"
        
        # 保存原始图像
        raw_path = self.save_dir / f"{base_name}_raw.jpg"
        cv2.imwrite(str(raw_path), raw_image)
        
        # 保存结果图像
        result_path = self.save_dir / f"{base_name}_result.jpg"
        cv2.imwrite(str(result_path), result_image)
        
        # 保存检测信息
        info_path = self.save_dir / f"{base_name}_info.txt"
        self._save_detection_info(info_path, detections, classes, timestamp)
        
        # 保存JSON格式（用于程序读取）
        json_path = self.save_dir / f"{base_name}.json"
        self._save_json(json_path, raw_image, detections, classes, timestamp)
        
        return {
            'raw': str(raw_path),
            'result': str(result_path),
            'info': str(info_path),
            'json': str(json_path)
        }
    
    def _save_detection_info(self, path: Path, detections: List[Detection], classes: List[str], timestamp: str):
        """保存检测信息到文本文件"""
        with open(path, 'w') as f:
            f.write(f"检测时间: {timestamp}\n")
            f.write(f"检测类别: {', '.join([c for c in classes if c])}\n")
            f.write(f"检测到 {len(detections)} 个目标:\n\n")
            
            for i, det in enumerate(detections, 1):
                f.write(f"目标 {i}:\n")
                f.write(f"  类别: {det.class_name}\n")
                f.write(f"  置信度: {det.confidence:.3f}\n")
                f.write(f"  边界框: {det.bbox.tolist()}\n\n")
    
    def _save_json(self, path: Path, image: np.ndarray, detections: List[Detection], classes: List[str], timestamp: str):
        """保存为JSON格式"""
        data = {
            'timestamp': timestamp,
            'image_shape': image.shape,
            'classes': classes,
            'detections': [
                {
                    'class_name': det.class_name,
                    'class_id': det.class_id,
                    'confidence': det.confidence,
                    'bbox': det.bbox.tolist()
                }
                for det in detections
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_history(self, limit: int = 5) -> List[Dict]:
        """加载历史记录"""
        history = []
        json_files = sorted(self.save_dir.glob("*.json"), reverse=True)
        
        for json_file in json_files[:limit]:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    history.append({
                        'timestamp': data.get('timestamp', ''),
                        'detections_count': len(data.get('detections', [])),
                        'classes': data.get('classes', []),
                        'file': str(json_file)
                    })
            except Exception as e:
                print(f"⚠️  无法读取 {json_file}: {e}")
        
        return history