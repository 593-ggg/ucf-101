import torch
import numpy as np
import pickle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import json
import matplotlib

matplotlib.use('TkAgg')  # 设置matplotlib后端
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 导入模型定义
from cnn_rnn import CNNRNN, VideoDataset
from cnn_3d import Simple3DCNN


class VideoActionRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("视频动作识别系统")
        self.root.geometry("1200x800")  # 增加宽度以容纳两个图表

        # 加载数据
        with open('processed_data/processed_data.pkl', 'rb') as f:
            data_dict = pickle.load(f)
        self.class_names = data_dict['class_names']
        self.test_data = data_dict['test_data']

        # 加载训练历史记录
        self.training_history = {}
        self.load_training_history()

        # 模型和状态
        self.models = {}
        self.current_model = None
        self.video_path = None
        self.video_frames = None

        # 设置matplotlib字体
        self.setup_matplotlib_font()

        # 设置界面
        self.setup_ui()

    def setup_matplotlib_font(self):
        """设置matplotlib字体，避免中文警告"""
        try:
            # 尝试使用系统字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

    def load_training_history(self):
        """加载训练历史记录"""
        try:
            if os.path.exists('training_history.json'):
                with open('training_history.json', 'r') as f:
                    self.training_history['CNN+RNN'] = json.load(f)
        except:
            self.training_history['CNN+RNN'] = None

        try:
            if os.path.exists('3dcnn_training_history.json'):
                with open('3dcnn_training_history.json', 'r') as f:
                    self.training_history['3D CNN'] = json.load(f)
        except:
            self.training_history['3D CNN'] = None

    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 模型选择
        ttk.Label(control_frame, text="选择模型:").grid(row=0, column=0, pady=5, sticky=tk.W)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var,
                                        values=["CNN+RNN", "3D CNN"], state="readonly")
        self.model_combo.grid(row=0, column=1, pady=5, padx=5)
        self.model_combo.current(0)

        # 加载模型按钮
        ttk.Button(control_frame, text="加载模型", command=self.load_model).grid(row=1, column=0, columnspan=2, pady=10)

        # 模型信息标签
        self.model_info_label = ttk.Label(control_frame, text="未加载模型", wraplength=200)
        self.model_info_label.grid(row=2, column=0, columnspan=2, pady=5)

        # 查看训练历史按钮
        ttk.Button(control_frame, text="查看训练历史", command=self.show_training_history).grid(row=3, column=0,
                                                                                                columnspan=2, pady=5)

        # 选择视频按钮
        ttk.Button(control_frame, text="选择视频文件", command=self.load_video_file).grid(row=4, column=0, columnspan=2,
                                                                                          pady=20)

        # 识别按钮
        ttk.Button(control_frame, text="识别动作", command=self.recognize_action,
                   style="Accent.TButton").grid(row=5, column=0, columnspan=2, pady=20)

        # 右侧显示区域
        display_frame = ttk.LabelFrame(main_frame, text="识别结果", padding="10")
        display_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 视频帧显示
        self.video_label = ttk.Label(display_frame, text="视频预览", relief="solid")
        self.video_label.grid(row=0, column=0, pady=5)

        # 结果标签
        self.result_label = ttk.Label(display_frame, text="识别结果: 等待识别", font=("Arial", 12))
        self.result_label.grid(row=1, column=0, pady=10)

        # 置信度显示
        self.confidence_label = ttk.Label(display_frame, text="置信度: -", font=("Arial", 10))
        self.confidence_label.grid(row=2, column=0, pady=5)

        # 识别状态显示 - 新添加的位置
        self.status_label = ttk.Label(display_frame, text="状态: 就绪", font=("Arial", 10))
        self.status_label.grid(row=3, column=0, pady=5)

        # 模型性能比较图表
        ttk.Label(display_frame, text="模型性能比较:", font=("Arial", 10, "bold")).grid(row=4, column=0, pady=10,
                                                                                        sticky=tk.W)

        # 创建准确率比较图表
        self.compare_fig = Figure(figsize=(5, 3), dpi=100)
        self.compare_ax = self.compare_fig.add_subplot(111)
        self.compare_ax.set_title('Model Accuracy Comparison')
        self.compare_ax.set_xlabel('Model')
        self.compare_ax.set_ylabel('Accuracy (%)')
        self.compare_ax.set_ylim([0, 100])

        self.compare_canvas = FigureCanvasTkAgg(self.compare_fig, master=display_frame)
        self.compare_canvas.draw()
        self.compare_canvas.get_tk_widget().grid(row=5, column=0, pady=5)

        # 中间训练历史显示区域
        history_frame = ttk.LabelFrame(main_frame, text="训练历史", padding="10")
        history_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 训练历史图表
        self.history_fig = Figure(figsize=(5, 3), dpi=100)
        self.history_ax = self.history_fig.add_subplot(111)
        self.history_ax.set_title('Training History')
        self.history_ax.set_xlabel('Epoch')
        self.history_ax.set_ylabel('Accuracy (%)')
        self.history_ax.set_ylim([0, 100])
        self.history_ax.grid(True, alpha=0.3)

        self.history_canvas = FigureCanvasTkAgg(self.history_fig, master=history_frame)
        self.history_canvas.draw()
        self.history_canvas.get_tk_widget().grid(row=0, column=0, pady=5)

        # 底部信息区域已移除，状态信息已移至display_frame中

    def show_training_history(self):
        """显示训练历史详情"""
        model_type = self.model_var.get()
        if not model_type:
            messagebox.showerror("错误", "请先选择模型类型")
            return

        history = self.training_history.get(model_type)
        if not history:
            messagebox.showinfo("提示", f"{model_type}模型的训练历史未找到")
            return

        # 创建训练历史详情窗口
        history_window = tk.Toplevel(self.root)
        history_window.title(f"{model_type} Training History")
        history_window.geometry("600x400")

        # 创建文本显示区域
        text_frame = ttk.Frame(history_window, padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True)

        text_widget = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 格式化显示训练历史
        text_widget.insert(tk.END, f"{model_type} Training History\n")
        text_widget.insert(tk.END, "=" * 50 + "\n\n")
        text_widget.insert(tk.END, f"Best Validation Accuracy: {history.get('best_val_acc', 0):.2f}%\n\n")
        text_widget.insert(tk.END, "Training Process:\n")
        for i, (train_loss, train_acc, val_acc) in enumerate(zip(
                history.get('train_loss', []),
                history.get('train_acc', []),
                history.get('val_acc', [])
        ), 1):
            text_widget.insert(tk.END, f"Epoch {i:2d}: Loss={train_loss:.4f}, "
                                       f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%\n")

        text_widget.config(state=tk.DISABLED)
        ttk.Button(history_window, text="关闭", command=history_window.destroy).pack(pady=10)
        self.update_training_history_chart(model_type)

    def update_training_history_chart(self, model_type):
        """更新训练历史图表"""
        self.history_ax.clear()

        history = self.training_history.get(model_type)
        if not history:
            return

        epochs = list(range(1, len(history.get('train_acc', [])) + 1))
        train_acc = history.get('train_acc', [])
        val_acc = history.get('val_acc', [])

        if epochs and train_acc and val_acc:
            self.history_ax.plot(epochs, train_acc, 'b-', label='Training Accuracy', marker='o', markersize=4)
            self.history_ax.plot(epochs, val_acc, 'r-', label='Validation Accuracy', marker='s', markersize=4)
            self.history_ax.legend()
            self.history_ax.set_title(f'{model_type} Training History')
            self.history_ax.set_xlabel('Epoch')
            self.history_ax.set_ylabel('Accuracy (%)')
            self.history_ax.set_ylim([0, 100])
            self.history_ax.grid(True, alpha=0.3)
            self.history_fig.tight_layout()
            self.history_canvas.draw()

    def load_model(self):
        model_type = self.model_var.get()
        if not model_type:
            messagebox.showerror("错误", "请先选择模型类型")
            return

        try:
            if model_type == "CNN+RNN":
                model = CNNRNN(num_classes=len(self.class_names))
                checkpoint = torch.load('best_cnn_rnn_model.pth', map_location='cpu')
            else:  # 3D CNN
                model = Simple3DCNN(num_classes=len(self.class_names))
                checkpoint = torch.load('best_3dcnn_model.pth', map_location='cpu')

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.models[model_type] = model
            self.current_model = model_type

            accuracy = checkpoint.get('val_acc', 0)
            self.model_info_label.config(text=f"{model_type}模型已加载\n准确率: {accuracy:.2f}%")
            self.status_label.config(text=f"{model_type}模型加载成功")

            # 更新图表
            self.update_accuracy_chart()
            self.update_training_history_chart(model_type)

        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {str(e)}")

    def load_video_file(self):
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("Video files", "*.avi *.mp4 *.mov"), ("All files", "*.*")]
        )

        if file_path:
            self.video_path = file_path
            self.status_label.config(text=f"已加载视频: {os.path.basename(file_path)}")

            # 提取视频文件名
            filename = os.path.basename(file_path)

            # 从文件名中提取真实标签
            # 假设文件名格式为: v_ApplyLipstick_g01_c01.avi
            # 提取动作类别（第二个下划线之前的部分）
            if '_' in filename:
                # 去掉扩展名
                filename_no_ext = os.path.splitext(filename)[0]
                # 按'_'分割
                parts = filename_no_ext.split('_')
                if len(parts) >= 2:
                    # 第一部分是'v'，第二部分是动作类别
                    action_class = parts[1]  # 比如 'ApplyLipstick'
                    # 找到对应的类别索引
                    if action_class in self.class_names:
                        self.true_label = self.class_names.index(action_class)
                        self.true_class = action_class
                        self.status_label.config(
                            text=f"已加载视频: {os.path.basename(file_path)} (真实标签: {self.true_class})")
                    else:
                        # 如果不在类别列表中，使用文件名
                        self.true_label = None
                        self.true_class = action_class
                else:
                    # 如果文件名格式不符合预期
                    self.true_label = None
                    self.true_class = None
            else:
                self.true_label = None
                self.true_class = None

            # 提取视频帧
            self.extract_video_frames(file_path)

            # 显示第一帧
            if self.video_frames is not None:
                self.display_video_frame(0)

    def extract_video_frames(self, video_path, num_frames=16):
        """从视频中提取帧"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < num_frames:
                messagebox.showwarning("警告", f"视频帧数不足 ({total_frames} < {num_frames})")
                cap.release()
                return

            # 均匀采样
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    frames.append(frame)

            cap.release()

            if len(frames) == num_frames:
                self.video_frames = np.array(frames)
                self.status_label.config(text=f"成功提取 {len(frames)} 帧")
            else:
                messagebox.showerror("错误", "帧提取失败")

        except Exception as e:
            messagebox.showerror("错误", f"视频处理失败: {str(e)}")

    def display_video_frame(self, frame_idx):
        """显示指定帧"""
        if self.video_frames is not None and 0 <= frame_idx < len(self.video_frames):
            frame = self.video_frames[frame_idx]
            img = Image.fromarray(frame)
            img = img.resize((300, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self.video_label.config(image=photo, text="")
            self.video_label.image = photo

    def recognize_action(self):
        if self.current_model is None or self.current_model not in self.models:
            messagebox.showerror("错误", "请先加载模型")
            return

        if self.video_frames is None:
            messagebox.showerror("错误", "请先加载视频")
            return

        try:
            self.status_label.config(text="正在识别...")
            self.root.update()

            # 预处理输入
            frames = self.video_frames.copy()
            frames_tensor = torch.FloatTensor(frames).permute(3, 0, 1, 2).unsqueeze(0) / 255.0

            # 使用模型进行预测
            model = self.models[self.current_model]
            with torch.no_grad():
                outputs = model(frames_tensor)

                # 使用softmax获取概率
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            predicted_class = self.class_names[predicted.item()]
            confidence_value = confidence.item() * 100  # 转换为百分比

            self.result_label.config(text=f"识别结果: {predicted_class}")
            self.confidence_label.config(text=f"置信度: {confidence_value:.2f}%")

            # 检查是否有真实标签
            if hasattr(self, 'true_label') and self.true_label is not None:
                # 从文件名中提取的真实标签
                if predicted_class == self.true_class:
                    self.status_label.config(text=f"识别正确! 真实标签: {self.true_class}")
                else:
                    self.status_label.config(text=f"识别错误! 预测: {predicted_class}, 真实: {self.true_class}")
            else:
                # 未知标签
                self.status_label.config(text="识别完成")

        except Exception as e:
            messagebox.showerror("错误", f"识别失败: {str(e)}")
            self.status_label.config(text="识别失败")

    def update_accuracy_chart(self):
        """更新准确率比较图表"""
        self.compare_ax.clear()

        accuracies = {}
        # 尝试加载两个模型的准确率
        try:
            if os.path.exists('best_cnn_rnn_model.pth'):
                checkpoint = torch.load('best_cnn_rnn_model.pth', map_location='cpu')
                accuracies['CNN+RNN'] = checkpoint.get('val_acc', 0)
        except:
            pass

        try:
            if os.path.exists('best_3dcnn_model.pth'):
                checkpoint = torch.load('best_3dcnn_model.pth', map_location='cpu')
                accuracies['3D CNN'] = checkpoint.get('val_acc', 0)
        except:
            pass

        if accuracies:
            models = list(accuracies.keys())
            values = list(accuracies.values())

            bars = self.compare_ax.bar(models, values, color=['skyblue', 'lightcoral'])

            # 在柱状图上添加数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                self.compare_ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                                     f'{value:.1f}%', ha='center', va='bottom')

            self.compare_ax.set_title('Model Accuracy Comparison')
            self.compare_ax.set_xlabel('Model')
            self.compare_ax.set_ylabel('Accuracy (%)')
            self.compare_ax.set_ylim([0, 100])
            self.compare_fig.tight_layout()
            self.compare_canvas.draw()


def main():
    root = tk.Tk()
    app = VideoActionRecognizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()