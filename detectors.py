import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from config import DOOR_MODEL_PATH, PERSON_DETECT_MODEL_PATH, PERSON_RECOGNIZE_MODEL_PATH
import os
class DoorStatusRecognizer:
    """门状态识别器"""
    def __init__(self, input_size: int = 224):
        self.ort_session = ort.InferenceSession(
            str(DOOR_MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )
        
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        self.input_size = input_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.class_map = {0: "close", 1: "open"}

    def _preprocess(self, image_frame: np.ndarray) -> np.ndarray:
        with Image.fromarray(image_frame).convert("RGB") as img:
            resize_size = int(self.input_size / 0.875)
            img = img.resize((resize_size, resize_size), Image.Resampling.LANCZOS)
            
            left = (resize_size - self.input_size) // 2
            top = (resize_size - self.input_size) // 2
            img = img.crop((left, top, left + self.input_size, top + self.input_size))
            
            img_np = np.array(img, dtype=np.float32) / 255.0
            img_np = (img_np - self.mean) / self.std
            img_np = img_np.transpose(2, 0, 1)
            img_np = np.expand_dims(img_np, axis=0)
            return img_np

    def predict(self, image_frame: np.ndarray) -> tuple:
        input_tensor = self._preprocess(image_frame)
        outputs = self.ort_session.run([self.output_name], {self.input_name: input_tensor})
        
        logits = outputs[0][0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        pred_idx = np.argmax(probabilities)
        
        if probabilities[pred_idx] > 0.980:
            pred_idx = 1
        else:
            pred_idx = 0
        
        return self.class_map[pred_idx], float(probabilities[pred_idx])


class PersonDetector:
    """人体检测器"""
    def __init__(self, conf_threshold=0.5, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = 640
        
        self.session = ort.InferenceSession(
            str(PERSON_DETECT_MODEL_PATH),
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self._shape_printed = False

    def preprocess(self, image):
        height, width = image.shape[:2]
        scale = self.input_size / max(width, height)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        
        padded_image = np.ones((self.input_size, self.input_size, 3), dtype=np.uint8) * 114
        pad_x = (self.input_size - new_width) // 2
        pad_y = (self.input_size - new_height) // 2
        padded_image[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized_image
        
        input_tensor = padded_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, scale, pad_x, pad_y

    def postprocess(self, outputs, scale, original_shape, pad_x=None, pad_y=None):
        boxes = []
        confidences = []
        class_ids = []
        
        output = outputs[0]
        if len(output.shape) == 3:
            output = output[0]
        
        for detection in output:
            confidence = detection[4]
            if confidence > self.conf_threshold:
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
                final_confidence = confidence * class_confidence
                
                if final_confidence > self.conf_threshold:
                    center_x = detection[0]
                    center_y = detection[1]
                    width = detection[2]
                    height = detection[3]
                    
                    if pad_x is not None and pad_y is not None:
                        center_x = center_x - pad_x
                        center_y = center_y - pad_y
                    
                    x = int((center_x - width / 2) / scale)
                    y = int((center_y - height / 2) / scale)
                    w = int(width / scale)
                    h = int(height / scale)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(final_confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return (
                [boxes[i] for i in indices],
                [confidences[i] for i in indices],
                [class_ids[i] for i in indices]
            )
        
        return [], [], []

    def detect(self, image):
        input_tensor, scale, pad_x, pad_y = self.preprocess(image)
        
        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            if not self._shape_printed:
                print(f"Model output shape: {[output.shape for output in outputs]}")
                self._shape_printed = True
        except Exception as e:
            print(f"Inference error: {e}")
            return [], [], []
        
        return self.postprocess(outputs, scale, image.shape[:2], pad_x, pad_y)


class PersonRecognizer:
    """人脸识别器"""
    def __init__(self, person_path):
        self.sess = ort.InferenceSession(
            str(PERSON_RECOGNIZE_MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )
        self.person_bank = self.init_person_bank(person_path)

    def init_person_bank(self, person_path):
        npy_dict = {}
        for filename in os.listdir(person_path):
            if filename.endswith(".npy"):            
                key = os.path.splitext(filename)[0]                
                filepath = os.path.join(person_path, filename)
                npy_dict[key] = np.load(filepath)
        return npy_dict

    def preprocess(self, img2, camid_src):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img2)
        img2 = img2.resize((128, 256), Image.BILINEAR)
        img2 = np.array(img2)
        img2 = img2.transpose(2, 0, 1)
        img2 = np.asarray((img2/255.0), dtype=np.float32)
        img2 = (img2 - 0.5)/0.5
        img2 = np.expand_dims(img2, 0)

        query_camid = np.array(camid_src, dtype=np.int64)
        query_camid = np.expand_dims(query_camid, 0)
        return img2, query_camid

    def infer(self, img_src, camid_src):
        img, query_camid = self.preprocess(img_src, camid_src)
        ort_inputs1 = {self.sess.get_inputs()[0].name: img, self.sess.get_inputs()[1].name: query_camid}
        output1 = self.sess.run(None, ort_inputs1)
        query_feat = output1[0]
        return query_feat

    @staticmethod
    def to_e(data_list):
        data = data_list[0]
        vector_e = data / np.sqrt(np.sum(np.power(data, 2)))
        return vector_e