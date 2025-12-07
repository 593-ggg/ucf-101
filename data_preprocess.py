import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import pickle


class VideoPreprocessor:
    def __init__(self, dataset_path='UCF-101', output_dir='processed_data',
                 target_fps=25, num_frames=16, img_size=(224, 224), num_classes=10):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.target_fps = target_fps
        self.num_frames = num_frames
        self.img_size = img_size
        self.num_classes = num_classes

        os.makedirs(output_dir, exist_ok=True)

    def get_video_paths(self):
        """获取前num_classes个类别的视频路径"""
        all_classes = sorted(os.listdir(self.dataset_path))
        selected_classes = all_classes[:self.num_classes]

        video_paths = []
        labels = []

        for label, class_name in enumerate(selected_classes):
            class_path = os.path.join(self.dataset_path, class_name)
            if os.path.isdir(class_path):
                videos = os.listdir(class_path)
                for video in videos:
                    video_paths.append(os.path.join(class_path, video))
                    labels.append(label)

        return video_paths, labels, selected_classes

    def extract_frames(self, video_path):
        """从视频中提取固定数量的帧"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if fps == 0:
            fps = self.target_fps

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < self.num_frames:
            return None

        # 计算采样间隔
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 调整大小和颜色空间
                frame = cv2.resize(frame, self.img_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        if len(frames) < self.num_frames:
            return None

        return np.array(frames)

    def preprocess_dataset(self, train_ratio=0.8):
        """预处理整个数据集"""
        print("开始数据预处理...")

        # 获取视频路径和标签
        video_paths, labels, class_names = self.get_video_paths()
        print(f"找到 {len(video_paths)} 个视频，{len(class_names)} 个类别")
        print(f"类别: {class_names}")

        # 打乱数据
        combined = list(zip(video_paths, labels))
        random.shuffle(combined)
        video_paths[:], labels[:] = zip(*combined)

        # 分割训练集和测试集
        split_idx = int(len(video_paths) * train_ratio)
        train_paths, test_paths = video_paths[:split_idx], video_paths[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]

        print(f"训练集: {len(train_paths)} 个视频")
        print(f"测试集: {len(test_paths)} 个视频")

        # 处理训练集
        train_data = []
        print("处理训练集视频...")
        for path, label in tqdm(zip(train_paths, train_labels), total=len(train_paths)):
            frames = self.extract_frames(path)
            if frames is not None:
                train_data.append((frames, label))

        # 处理测试集
        test_data = []
        print("处理测试集视频...")
        for path, label in tqdm(zip(test_paths, test_labels), total=len(test_paths)):
            frames = self.extract_frames(path)
            if frames is not None:
                test_data.append((frames, label))

        # 保存数据
        data_dict = {
            'train_data': train_data,
            'test_data': test_data,
            'class_names': class_names
        }

        with open(os.path.join(self.output_dir, 'processed_data.pkl'), 'wb') as f:
            pickle.dump(data_dict, f)

        print(f"数据预处理完成！保存到 {os.path.join(self.output_dir, 'processed_data.pkl')}")

        return data_dict


if __name__ == "__main__":
    preprocessor = VideoPreprocessor(dataset_path='UCF-101')
    data = preprocessor.preprocess_dataset()