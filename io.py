import os
import time
from config import COMM_MSG_FILE, LOCK_FILE

def send_message(msg):
    """进程间通信：发送指令"""
    while os.path.exists(LOCK_FILE):
        time.sleep(0.1)
    
    with open(LOCK_FILE, 'w') as f:
        f.write("locked")
    
    try:
        with open(COMM_MSG_FILE, 'w') as f:
            f.write(msg)
    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)

def get_message():
    """获取通信消息"""
    if not os.path.exists(COMM_MSG_FILE) or os.path.exists(LOCK_FILE):
        return None
    
    try:
        with open(COMM_MSG_FILE, 'r') as f:
            return f.read().strip()
    except:
        return None

def clear_temp_files(temp_dir):
    """清空临时文件"""
    if not os.path.exists(temp_dir):
        return
    
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"清除文件失败 {file_path}: {e}")