import cv2
import sys
import os
import time
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from .ui.layout import MainLayout
from .core import DoorStatusRecognizer, FrameDrawer, ResultFilter, PersonDetectionProcessor2
from .utils import send_message, clear_temp_files
from config import (
    BASE_WIDTH, BASE_HEIGHT, VIDEO_RENDER_WIDTH, VIDEO_RENDER_HEIGHT,
    FRAME_SKIP, TEMP_DIR, VIDEO_PATH, LOG_CSV_PATH
)

def init_log_dir():
    """初始化日志目录和文件"""
    log_dir = os.path.dirname(LOG_CSV_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    if not os.path.exists(LOG_CSV_PATH):
        with open(LOG_CSV_PATH, 'w', encoding='utf-8', newline='') as f:
            f.write("日期,时间,人员列表,楼层,进/出\n")
    print(f"日志文件初始化完成：{LOG_CSV_PATH}")

class ElevatorSecurityUI(tk.Tk):
    """电梯安防系统主界面"""
    def __init__(self):
        super().__init__()
        self.title("梯控天眼-智慧安防平台")
        self.geometry(f"{BASE_WIDTH}x{BASE_HEIGHT}")
        self.resizable(True, True)
        self.minsize(int(BASE_WIDTH * 0.8), int(BASE_HEIGHT * 0.8))
        
        self.last_log_line = 0
        self.video_running = True
        self.current_frame = None
        self.person_images = {}
        self.door_status = "closed"
        self.door_confidence = 0.0
        self.frame_count = 0
        self.last_door_state = "closed"
        
        init_log_dir()
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # 初始化布局
        self.layout = MainLayout(self)
        
        # 初始化核心处理器
        self._init_core_processors()
        
        # 启动后台线程
        self._start_background_threads()
        
        # 启动人员检测处理器
        self.person_processor = PersonDetectionProcessor2()
        self.processor_thread = threading.Thread(target=self.person_processor.run, daemon=True)
        self.processor_thread.start()

    def _init_core_processors(self):
        """初始化核心处理器"""
        try:
            self.door_recognizer = DoorStatusRecognizer()
            self.frame_drawer = FrameDrawer()
            self.result_filter = ResultFilter()
            print("门状态识别模型加载成功")
        except Exception as e:
            messagebox.showerror("初始化失败", f"门状态识别模型加载失败：{str(e)}")
            sys.exit(1)
        
        try:
            self.cap = cv2.VideoCapture(str(VIDEO_PATH))
            if not self.cap.isOpened():
                raise Exception(f"无法打开视频文件：{VIDEO_PATH}")
            print(f"视频源加载成功：{VIDEO_PATH}")
        except Exception as e:
            messagebox.showerror("初始化失败", f"视频源加载失败：{str(e)}")
            sys.exit(1)

    def _start_background_threads(self):
        """启动后台线程"""
        self.video_thread = threading.Thread(target=self._process_video, daemon=True)
        self.video_thread.start()
        
        self.update_thread = threading.Thread(target=self._update_ui, daemon=True)
        self.update_thread.start()

    def _process_video(self):
        """处理视频帧"""
        while self.video_running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            self.frame_count += 1
            if self.frame_count % (FRAME_SKIP + 1) != 0:
                continue
            
            # 检测门状态
            door_status, confidence = self.door_recognizer.predict(frame)
            self.door_status = door_status
            self.door_confidence = confidence
            
            # 门状态变化处理
            if self.door_status != self.last_door_state:
                if self.door_status == "open":
                    send_message("START_DETECTION")
                    self.layout.clear_recognized_persons()
                else:
                    send_message("STOP_DETECTION")
                self.last_door_state = self.door_status
            
            # 保存帧用于处理
            if self.door_status == "open":
                frame_path = os.path.join(TEMP_DIR, f"frame_{time.time()}.jpg")
                cv2.imwrite(frame_path, frame)
            
            # 绘制信息
            info = {
                "时间": time.strftime("%H:%M:%S"),
                "门状态": f"{self.door_status} ({self.door_confidence:.2f})",
                "帧计数": self.frame_count
            }
            
            processed_frame = self.frame_drawer.draw_info(frame, info, status_key="门状态")
            
            # 缩放用于显示
            resized_frame = cv2.resize(processed_frame, (VIDEO_RENDER_WIDTH, VIDEO_RENDER_HEIGHT))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            self.current_frame = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
            
            # 更新门状态显示
            self.layout.update_door_status(self.door_status, self.door_confidence)
            
            time.sleep(0.01)

    def _update_ui(self):
        """更新UI"""
        while True:
            if self.current_frame:
                self.layout.update_video_frame(self.current_frame)
            
            time.sleep(0.05)

    def on_close(self):
        """关闭窗口处理"""
        self.video_running = False
        send_message("EXIT")
        self.person_processor.running = False
        self.cap.release()
        clear_temp_files(TEMP_DIR)
        self.destroy()

def main():
    app = ElevatorSecurityUI()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

if __name__ == "__main__":
    main()