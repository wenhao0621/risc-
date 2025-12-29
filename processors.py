import os
import time
import cv2
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import TEMP_DIR, LOG_CSV_PATH, PERSON_IMAGE_DIR, COMM_MSG_FILE, LOCK_FILE
from .detectors import PersonDetector, PersonRecognizer

class FrameDrawer:
    """帧图像绘制器"""
    def __init__(self, 
                 text_color=(255, 255, 255),
                 font_scale=0.6,
                 thickness=1,
                 margin=10,
                 line_spacing=5):
        self.text_color = text_color
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
        self.margin = margin
        self.line_spacing = line_spacing

    def draw_info(self, frame, info, vertices=None, status_key=None):
        frame_copy = frame.copy()
        lines = [f"{k}: {v}" for k, v in info.items()]
        
        max_width, total_height, text_sizes = self._calculate_text_area(lines)
        overlay = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
        
        self._draw_text(overlay, lines, text_sizes, status_key)
        
        if vertices is not None and len(vertices) > 0:
            self._draw_polygon(overlay, vertices)
            
        alpha = overlay[:, :, 3] / 255.0
        for c in range(0, 3):
            frame_copy[:, :, c] = (alpha * overlay[:, :, c] + 
                                  (1 - alpha) * frame_copy[:, :, c])
            
        return frame_copy

    def _calculate_text_area(self, lines):
        max_width = 0
        total_height = 0
        text_sizes = []
        
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, self.font, self.font_scale, self.thickness)
            text_sizes.append((w, h))
            max_width = max(max_width, w)
            total_height += h + self.line_spacing
            
        return max_width, total_height, text_sizes

    def _draw_text(self, overlay, lines, text_sizes, status_key):
        current_y = self.margin + self.margin + text_sizes[0][1]
        
        for i, (line, (w, h)) in enumerate(zip(lines, text_sizes)):
            color = self.text_color
            if status_key and lines[i].startswith(f"{status_key}:"):
                color = (0, 255, 0) if "open" in line.lower() else (0, 0, 255)
            
            cv2.putText(overlay, line, (self.margin + self.margin, current_y),
                       self.font, self.font_scale, (*color, 255), self.thickness)
            
            x1 = self.margin
            y1 = current_y - h - self.line_spacing // 2
            x2 = x1 + w + self.margin * 2
            y2 = current_y + self.line_spacing // 2
            overlay[y1:y2, x1:x2, 3] = 255
            current_y += h + self.line_spacing

    def _draw_polygon(self, overlay, vertices):
        cv2.polylines(overlay, [vertices.astype(np.int32)], 
                     True, (*(0, 0, 255), 255), 2)
        cv2.fillPoly(overlay, [vertices.astype(np.int32)], (0, 0, 0, 255))


class ResultFilter:
    """结果过滤器"""
    def __init__(self, min_confidence=0.6, smooth_window=3):
        self.min_confidence = min_confidence
        self.smooth_window = smooth_window
        self.history = []

    def filter(self, results):
        self.history.append(results)
        if len(self.history) > self.smooth_window:
            self.history.pop(0)
            
        filtered = {}
        for frame_results in self.history:
            for name, conf in frame_results.items():
                if name not in filtered or conf > filtered[name]:
                    filtered[name] = conf
                    
        return {k: v for k, v in filtered.items() if v >= self.min_confidence}


class PersonDetectionProcessor:
    """人体检测处理器"""
    def __init__(self):
        self.detector = PersonDetector()
        self.recognizer = PersonRecognizer(str(PERSON_IMAGE_DIR))
        
    def person_detection_and_recognition(self, frame):
        boxes, _, _ = self.detector.detect(frame)
        results = {}
        
        for i, (x, y, w, h) in enumerate(boxes):
            person_img = frame[y:y+h, x:x+w]
            if person_img.size == 0:
                continue
                
            try:
                feat = self.recognizer.infer(person_img, 0)
                feat = self.recognizer.to_e(feat)
                
                max_sim = -1
                best_match = f"unknown_{i}"
                
                for name, bank_feat in self.recognizer.person_bank.items():
                    sim = np.dot(feat, bank_feat)
                    if sim > max_sim and sim > 0.5:
                        max_sim = sim
                        best_match = name
                        
                results[best_match] = person_img
            except Exception as e:
                print(f"识别出错: {e}")
                results[f"unknown_{i}"] = person_img
                
        return results


class PersonDetectionProcessor2:
    """人体检测处理线程"""
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = True
        self.processing = False
        self.processed_files = set()
        self.detection_active = False
        self.processor = PersonDetectionProcessor()
        self.current_session_results = set()
        self.last_session_results = set()

    def write_log(self, known_persons, status_dict):
        current_date = time.strftime("%Y-%m-%d", time.localtime())
        current_time = time.strftime("%H:%M:%S", time.localtime())
        
        name_list_str = ",".join(known_persons) if known_persons else ""
        floor = "11"
        status = status_dict[known_persons[0]] if known_persons else ""
        
        log_line = f"{current_date},{current_time},{name_list_str},{floor},{status}\n"
        
        with open(LOG_CSV_PATH, "a+", encoding="utf-8", newline="") as f:
            f.seek(0)
            if not f.read():
                f.write("日期,时间,人员列表,楼层,进/出\n")
            f.write(log_line)
        print(f"日志已追加到 {LOG_CSV_PATH}")

    def get_message(self):
        if not os.path.exists(COMM_MSG_FILE) or os.path.exists(LOCK_FILE):
            return None
        try:
            with open(COMM_MSG_FILE, 'r') as f:
                return f.read().strip()
        except:
            return None
    
    def process_single_frame(self, frame_path):
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                return f"无法读取图像: {frame_path}"
            
            results = self.processor.person_detection_and_recognition(frame)
            name_list = []
            for name, person_img in results.items():
                name_list.append(name)
                if self.detection_active:
                    self.current_session_results.add(name)
            return name_list
        except Exception as e:
            return f"处理失败 {frame_path}: {str(e)}"
    
    def print_session_summary(self):
        if not self.current_session_results:
            print("本次开门期间未识别到任何人")
            self.last_session_results = set()
            return
            
        known_persons = [name for name in self.current_session_results if not name.startswith("unknown_")]
        unknown_persons = [name for name in self.current_session_results if name.startswith("unknown_")]
        total_persons = len(known_persons) + len(unknown_persons)
        
        status_dict = {}
        for person in known_persons:
            if person not in self.last_session_results:
                status_dict[person] = "进"
            else:
                status_dict[person] = "出"
        
        print(f"\n===== 本次开门期间识别结果 =====")
        print(f"总识别人数: {total_persons}")
        if known_persons:
            print(f"已知人员: {', '.join(known_persons)}")
            print(f"人员状态: {status_dict}")
        else:
            print("已知人员: 无")
        print(f"未知人员数量: {len(unknown_persons)}")
        print("================================\n")
        
        if known_persons:
            self.write_log(known_persons, status_dict)
        
        self.last_session_results = self.current_session_results.copy()
        self.current_session_results.clear()
    
    def process_batch(self):
        if not os.path.exists(TEMP_DIR):
            return
            
        frame_files = sorted(glob.glob(os.path.join(TEMP_DIR, "*.jpg")),
                            key=lambda x: os.path.getctime(x))
        if not frame_files:
            return
            
        print(f"开始批量处理，共{len(frame_files)}张图像")
        self.processing = True
        
        futures = [self.executor.submit(self.process_single_frame, fp) for fp in frame_files]
        
        for future in as_completed(futures):
            name_list = future.result()
            print(name_list)
        
        self.processing = False
        print("批量处理完成")

    def process_new_frames(self):
        frame_files = sorted(glob.glob(os.path.join(TEMP_DIR, "*.jpg")),
                            key=lambda x: os.path.getctime(x))
        new_files = [f for f in frame_files if f not in self.processed_files]
        
        if new_files:
            print(f"发现{len(new_files)}个新图像，开始处理...")
            futures = [self.executor.submit(self.process_single_frame, fp) for fp in new_files]
            for future in as_completed(futures):
                name_list = future.result()
                print(name_list)
            self.processed_files.update(new_files)

    def clear_temp_files(self):
        if self.processing:
            return False 
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"清除文件失败 {file_path}: {e}")
        return True
    
    def run(self):
        print("人体检测识别程序启动，等待指令...")
        while self.running:
            msg = self.get_message()
            if msg == "START_DETECTION":
                self.detection_active = True
                self.current_session_results.clear()
                print("开始检测模式，持续处理新增图像...")
            elif msg == "STOP_DETECTION":
                self.detection_active = False
                print("停止检测模式，等待所有处理完成...")
                while self.processing:
                    time.sleep(0.05)
                self.process_new_frames()
                self.print_session_summary()
                self.clear_temp_files()
                print("已清空临时文件")
            elif msg == "EXIT":
                self.running = False
                self.detection_active = False
                print("准备退出程序...")
                
            if self.detection_active:
                self.processing = True
                self.process_new_frames()
                self.processing = False

        print("程序退出")