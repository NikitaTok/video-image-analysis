import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import threading
import time
from collections import deque
from io import BytesIO

class HistogramAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор гистограмм")
        self.root.geometry("1200x800")
        
        self.image = None
        self.video_capture = None
        self.is_playing = False
        self.video_thread = None
        self.frame_queue = deque(maxlen=2)
        self.enhanced_frame_queue = deque(maxlen=2)
        self.last_frame_time = 0
        self.frame_interval = 1/60
        self.progress_updater = None
        self.video_paused = False
        self.seeking = False
        
        self.histogram_captured = False
        self.captured_histogram = None
        self.captured_frame = None
        
        self.before_histogram_data = None
        self.after_histogram_data = None
        self.before_strobe_histograms = None
        self.after_strobe_histograms = None
        
        self.current_photo = None
        self.current_enhanced_photo = None
        self.current_hist_photo = None
        self._hist_update_time = 0
        self._hist_buffer = np.zeros(32, dtype=np.float32)
        
        self.progress_update_paused = False
        self._last_seek_time = 0
        self._last_seek_frame = 0
        self.current_progress = 0
        
        self.current_frame_stats = {
            'mean': 0,
            'variance': 0,
            'std': 0,
            'frame_number': 0,
            'width': 0,
            'height': 0
        }
        
        self.show_strobe = False
        self.strobe_rect = None
        self.strobe_thickness = 4
        self.strobe_color = (0, 255, 0)
        
        self.enhancement_method = None
        self.apply_enhancement = False
        
        self.enhancement_params = {
            'current_peak': 0,
            'black': 0,
            'white': 0,
            'gamma': 1.0
        }
        
        self.manual_params = {
            'black': tk.IntVar(value=0),
            'white': tk.IntVar(value=255),
            'gamma': tk.DoubleVar(value=1.0)
        }
        
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
        
        self.setup_ui()
    
    def setup_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        display_frame = tk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Button(control_frame, text="Загрузить изображение", command=self.load_image, width=20).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Загрузить видео", command=self.load_video, width=20).pack(side=tk.LEFT, padx=5)
        
        self.strobe_button = tk.Button(control_frame, text="Область", command=self.toggle_strobe, width=15, state=tk.DISABLED)
        self.strobe_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(control_frame, text="Стоп", command=self.stop_video, width=15, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.histogram_mode_var = tk.StringVar()
        self.histogram_mode_var.set("Выберите режим")
        
        self.histogram_mode_combo = ttk.Combobox(control_frame, textvariable=self.histogram_mode_var, width=25, state="readonly")
        self.histogram_mode_combo['values'] = ("Гистограмма", "Гистограмма фон+объект", "Гистограмма до/после", "Гистограмма фон+объект до/после")
        self.histogram_mode_combo.pack(side=tk.LEFT, padx=5)
        self.histogram_mode_combo.bind("<<ComboboxSelected>>", self.on_histogram_mode_selected)
        
        self.apply_histogram_button = tk.Button(control_frame, text="Применить гист.", command=self.apply_histogram_mode, width=12, state=tk.DISABLED)
        self.apply_histogram_button.pack(side=tk.LEFT, padx=5)
        
        self.enhancement_mode_var = tk.StringVar()
        self.enhancement_mode_var.set("Методы улучшения")
        
        self.enhancement_mode_combo = ttk.Combobox(control_frame, textvariable=self.enhancement_mode_var, width=15, state="readonly")
        self.enhancement_mode_combo['values'] = ("Без улучшения", "Levels", "Gause+CLAHE", "Levels параметры")
        self.enhancement_mode_combo.pack(side=tk.LEFT, padx=5)
        self.enhancement_mode_combo.bind("<<ComboboxSelected>>", self.on_enhancement_mode_selected)
        
        self.image_frame = tk.Frame(display_frame, bg='white', relief=tk.SUNKEN, bd=2)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.enhanced_frame = tk.Frame(display_frame, bg='white', relief=tk.SUNKEN, bd=2)
        self.enhanced_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.hist_frame = tk.Frame(display_frame, bg='white', relief=tk.SUNKEN, bd=2)
        self.hist_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.image_label = tk.Label(self.image_frame, text="Исходное видео", bg='white')
        self.image_label.pack(expand=True)
        
        self.enhanced_label = tk.Label(self.enhanced_frame, text="Улучшенное видео", bg='white')
        self.enhanced_label.pack(expand=True)
        
        self.hist_label = tk.Label(self.hist_frame, text="Гистограмма не захвачена", bg='white')
        self.hist_label.pack(expand=True)
        
        self.stats_frame = tk.Frame(self.root, bg='lightgray', relief=tk.RAISED, bd=1)
        self.stats_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        self.stats_label = tk.Label(self.stats_frame, text="Статистика: Среднее: -- | Дисперсия: -- | STD: -- | Кадр: -- | Размер: --x--", bg='lightgray', anchor=tk.W, justify=tk.LEFT)
        self.stats_label.pack(fill=tk.X, padx=5, pady=2)
        
        self.levels_stats_label = tk.Label(self.stats_frame, text="Параметры Levels: Peak: -- | Black: -- | White: -- | Gamma: --", bg='lightgray', anchor=tk.W, justify=tk.LEFT)
        self.levels_stats_label.pack(fill=tk.X, padx=5, pady=2)
        
        self.params_frame = tk.Frame(self.root, bg='lightblue', relief=tk.RAISED, bd=1)
        self.params_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5, before=self.stats_frame)
        self.params_frame.pack_forget()
        
        self.create_manual_controls()
        
        self.info_label = tk.Label(self.root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.info_label.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.setup_progress_bar_and_navigation()
    
    def create_manual_controls(self):
        params_title = tk.Label(self.params_frame, text="Ручные параметры Levels:", bg='lightblue', font=('Arial', 10, 'bold'))
        params_title.pack(anchor=tk.W, padx=5, pady=2)
        
        sliders_frame = tk.Frame(self.params_frame, bg='lightblue')
        sliders_frame.pack(fill=tk.X, padx=10, pady=5)
        
        black_frame = tk.Frame(sliders_frame, bg='lightblue')
        black_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(black_frame, text="Black:", bg='lightblue', width=8).pack(side=tk.LEFT)
        black_slider = tk.Scale(black_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.manual_params['black'], length=200, command=self.on_manual_param_change)
        black_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        black_value = tk.Label(black_frame, textvariable=self.manual_params['black'], bg='lightblue', width=4)
        black_value.pack(side=tk.LEFT)
        
        white_frame = tk.Frame(sliders_frame, bg='lightblue')
        white_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(white_frame, text="White:", bg='lightblue', width=8).pack(side=tk.LEFT)
        white_slider = tk.Scale(white_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.manual_params['white'], length=200, command=self.on_manual_param_change)
        white_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        white_value = tk.Label(white_frame, textvariable=self.manual_params['white'], bg='lightblue', width=4)
        white_value.pack(side=tk.LEFT)
        
        gamma_frame = tk.Frame(sliders_frame, bg='lightblue')
        gamma_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(gamma_frame, text="Gamma:", bg='lightblue', width=8).pack(side=tk.LEFT)
        gamma_slider = tk.Scale(gamma_frame, from_=0.1, to=3.0, orient=tk.HORIZONTAL, variable=self.manual_params['gamma'], length=200, resolution=0.05, command=self.on_manual_param_change)
        gamma_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        gamma_value = tk.Label(gamma_frame, textvariable=self.manual_params['gamma'], bg='lightblue', width=6)
        gamma_value.pack(side=tk.LEFT)
        
        reset_button = tk.Button(self.params_frame, text="Сбросить к авто", command=self.reset_to_auto_params, width=15)
        reset_button.pack(pady=5)
    
    def on_manual_param_change(self, *args):
        if self.enhancement_method == "Levels параметры" and self.apply_enhancement:
            self.update_both_images()
            self.update_levels_stats_display()
            
            black = self.manual_params['black'].get()
            white = self.manual_params['white'].get()
            gamma = self.manual_params['gamma'].get()
            
            self.update_info(f"Ручные параметры Levels: Black={black}, White={white}, Gamma={gamma:.2f}")
    
    def reset_to_auto_params(self):
        if self.image is not None:
            if len(self.image.shape) == 3:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.image
            
            peak_value = self.calculate_histogram_peak(gray)
            black, white, gamma = self.calculate_levels_params(peak_value)
            
            self.manual_params['black'].set(int(black))
            self.manual_params['white'].set(int(white))
            self.manual_params['gamma'].set(float(gamma))
            
            self.update_info(f"Параметры сброшены к автоматическим: Black={black:.0f}, White={white:.0f}, Gamma={gamma:.2f}")
    
    def on_enhancement_mode_selected(self, event=None):
        selected_method = self.enhancement_mode_var.get()
        
        if selected_method == "Методы улучшения":
            return
        
        self.enhancement_method = selected_method
        self.apply_enhancement = (selected_method != "Без улучшения")
        
        if selected_method == "Levels параметры":
            self.params_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5, before=self.stats_frame)
            if self.image is not None:
                self.reset_to_auto_params()
        else:
            self.params_frame.pack_forget()
        
        if self.apply_enhancement:
            self.update_info(f"Применяется метод улучшения: {selected_method}")
            
            self.after_histogram_data = None
            self.after_strobe_histograms = None
            
            current_hist_mode = self.histogram_mode_var.get()
            if current_hist_mode in ["Гистограмма до/после", "Гистограмма фон+объект до/после"]:
                self.apply_histogram_mode()
        else:
            self.update_info("Улучшение отключено")
        
        self.enhanced_frame_queue.clear()
        
        if self.video_capture and self.is_playing:
            current_frame = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            
        self.update_both_images()
    
    def apply_enhancement_to_frame(self, frame):
        if not self.apply_enhancement or frame is None:
            return frame
        
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        if self.enhancement_method == "Levels":
            return self.apply_levels_to_frame(gray)
        elif self.enhancement_method == "Gause+CLAHE":
            return self.apply_gause_clahe_to_frame(gray)
        elif self.enhancement_method == "Levels параметры":
            return self.apply_manual_levels_to_frame(gray)
        
        return gray
    
    def apply_manual_levels_to_frame(self, gray_frame):
        black = self.manual_params['black'].get()
        white = self.manual_params['white'].get()
        gamma = self.manual_params['gamma'].get()
        
        peak_value = self.calculate_histogram_peak(gray_frame)
        self.enhancement_params = {
            'current_peak': peak_value,
            'black': black,
            'white': white,
            'gamma': gamma
        }
        
        levels_result = gray_frame.copy().astype(np.float32)
        levels_result = np.maximum(levels_result - black, 0)
        
        if white > black:
            levels_result = levels_result * (255.0 / (white - black))
        else:
            levels_result = levels_result * 255.0
            
        levels_result = np.clip(levels_result, 0, 255)
        
        if gamma != 0:
            levels_result = 255 * np.power(levels_result / 255.0, 1/gamma)
        
        levels_result = levels_result.astype(np.uint8)
        
        return levels_result
    
    def apply_levels_to_frame(self, gray_frame):
        peak_value = self.calculate_histogram_peak(gray_frame)
        black, white, gamma = self.calculate_levels_params(peak_value)
        
        self.enhancement_params = {
            'current_peak': peak_value,
            'black': black,
            'white': white,
            'gamma': gamma
        }
        
        levels_result = gray_frame.copy().astype(np.float32)
        levels_result = np.maximum(levels_result - black, 0)
        levels_result = levels_result * (255.0 / (white - black))
        levels_result = np.clip(levels_result, 0, 255)
        levels_result = 255 * np.power(levels_result / 255.0, 1/gamma)
        levels_result = levels_result.astype(np.uint8)
        
        return levels_result
    
    def apply_gause_clahe_to_frame(self, gray_frame, peak_value=None):
        if len(gray_frame.shape) == 3:
            if gray_frame.shape[2] == 3:
                gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
            elif gray_frame.shape[2] == 4:
                gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGRA2GRAY)
            else:
                gray_frame = gray_frame[:, :, 0]
        
        if peak_value is None:
            peak_value = self.calculate_histogram_peak(gray_frame)
        
        levels_result = self.apply_levels_to_frame(gray_frame)
        denoised_gaussian = cv2.GaussianBlur(levels_result, (3, 3), 0)
        clahe_applied = self.clahe.apply(denoised_gaussian)
        
        return clahe_applied
    
    def calculate_histogram_peak(self, gray_image):
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        peak_value = np.argmax(hist)
        return peak_value
    
    def calculate_levels_params(self, peak_value):
        if peak_value <= 70:
            black = peak_value / 1.4
            white = peak_value * 1.84
            gamma = 1.35
        elif 71 < peak_value <= 102:
            black = peak_value - 20
            white = peak_value * 1.8
            gamma = 1.43
        else:
            black = peak_value - 18
            white = peak_value * 1.84
            gamma = 1.55
        
        black = max(0, min(black, 255))
        white = max(0, min(white, 255))
        
        if white <= black:
            white = min(255, black + 10)
        
        return black, white, gamma
    
    def update_levels_stats_display(self):
        params = self.enhancement_params
        
        if self.enhancement_method == "Levels параметры":
            stats_text = (f"Параметры Levels (ручные): Peak: {params['current_peak']:.1f} | "
                         f"Black: {self.manual_params['black'].get()} | "
                         f"White: {self.manual_params['white'].get()} | "
                         f"Gamma: {self.manual_params['gamma'].get():.2f}")
        else:
            stats_text = (f"Параметры Levels: Peak: {params['current_peak']:.1f} | "
                         f"Black: {params['black']:.1f} | "
                         f"White: {params['white']:.1f} | "
                         f"Gamma: {params['gamma']:.2f}")
        
        try:
            if self.levels_stats_label.winfo_exists():
                self.levels_stats_label.config(text=stats_text)
        except tk.TclError:
            pass
    
    def toggle_strobe(self):
        if self.image is None and self.video_capture is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение или видео")
            return
        
        self.show_strobe = not self.show_strobe
        
        if self.show_strobe:
            self.strobe_button.config(fg="black")
            self.update_info("Область (строб) включена")
            
            if self.strobe_rect is None:
                self.calculate_strobe_rect()
        else:
            self.strobe_button.config(fg="black")
            self.update_info("Область (строб) выключена")
        
        self.frame_queue.clear()
        self.enhanced_frame_queue.clear()
        
        if self.image is not None:
            self.update_both_images()
    
    def calculate_strobe_rect(self):
        if self.image is not None:
            height, width = self.image.shape[:2]
        elif self.video_capture and self.is_playing:
            width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            return
        
        scale_factor = 4 # строб в 4 раза меньше
        new_width = int(width / scale_factor)
        new_height = int(height / scale_factor)
        
        x1 = (width - new_width) // 2
        y1 = (height - new_height) // 2
        x2 = x1 + new_width
        y2 = y1 + new_height
        
        self.strobe_rect = [x1, y1, x2, y2]
        
        self.update_info(f"Строб: {new_width}x{new_height} пикселей (центр изображения)")
    
    def apply_strobe_to_frame(self, frame):
        if not self.show_strobe or self.strobe_rect is None:
            return frame
        
        frame_with_strobe = frame.copy()
        
        if len(frame_with_strobe.shape) == 2:
            frame_with_strobe = cv2.cvtColor(frame_with_strobe, cv2.COLOR_GRAY2BGR)
        elif len(frame_with_strobe.shape) == 3 and frame_with_strobe.shape[2] == 1:
            frame_with_strobe = cv2.cvtColor(frame_with_strobe, cv2.COLOR_GRAY2BGR)
        
        x1, y1, x2, y2 = self.strobe_rect
        cv2.rectangle(frame_with_strobe, (x1, y1), (x2, y2), self.strobe_color, self.strobe_thickness)
        
        strobe_width = x2 - x1
        strobe_height = y2 - y1
        text = f"Область: {strobe_width}x{strobe_height}"
        cv2.putText(frame_with_strobe, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.strobe_color, 1)
        
        return frame_with_strobe
    
    def extract_strobe_areas(self, frame):
        if self.strobe_rect is None:
            return None, None
        
        x1, y1, x2, y2 = self.strobe_rect
        
        inner_area = frame[y1:y2, x1:x2].copy()
        
        height, width = frame.shape[:2]
        mask = np.ones((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 0
        
        if len(frame.shape) == 3:
            background = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            background = cv2.bitwise_and(frame, frame, mask=mask)
        
        return inner_area, background
    
    def calculate_strobe_histograms(self, frame):
        if frame is None:
            return None, None
        
        inner_area, background_area = self.extract_strobe_areas(frame)
        
        if inner_area is None or background_area is None:
            return None, None
        
        if len(inner_area.shape) == 3:
            inner_gray = cv2.cvtColor(inner_area, cv2.COLOR_BGR2GRAY)
        else:
            inner_gray = inner_area
        
        if len(background_area.shape) == 3:
            background_gray = cv2.cvtColor(background_area, cv2.COLOR_BGR2GRAY)
        else:
            background_gray = background_area
        
        background_nonzero_mask = (background_gray > 0).astype(np.uint8)
        
        hist_inner = self.fast_histogram(inner_gray, bins=128)
        
        if np.sum(background_nonzero_mask) > 0:
            background_nonzero = background_gray[background_gray > 0]
            hist_background = self.fast_histogram(background_nonzero, bins=128)
        else:
            hist_background = np.zeros(128, dtype=np.float32)
        
        if hist_inner.max() > 0:
            hist_inner = hist_inner / hist_inner.max()
        if hist_background.max() > 0:
            hist_background = hist_background / hist_background.max()
        
        return hist_inner, hist_background
    
    def calculate_strobe_stats(self, frame):
        if frame is None or self.strobe_rect is None:
            return None, None
        
        inner_area, background_area = self.extract_strobe_areas(frame)
        
        if inner_area is None or background_area is None:
            return None, None
        
        if len(inner_area.shape) == 3:
            inner_gray = cv2.cvtColor(inner_area, cv2.COLOR_BGR2GRAY)
        else:
            inner_gray = inner_area
        
        if len(background_area.shape) == 3:
            background_gray = cv2.cvtColor(background_area, cv2.COLOR_BGR2GRAY)
        else:
            background_gray = background_area
        
        background_nonzero = background_gray[background_gray > 0]
        
        if inner_gray.size > 0:
            object_stats = {
                'mean': np.mean(inner_gray),
                'variance': np.var(inner_gray),
                'std': np.std(inner_gray),
                'min': np.min(inner_gray),
                'max': np.max(inner_gray),
                'median': np.median(inner_gray),
                'pixel_count': inner_gray.size
            }
        else:
            object_stats = {'mean': 0, 'variance': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0, 'pixel_count': 0}
        
        if background_nonzero.size > 0:
            background_stats = {
                'mean': np.mean(background_nonzero),
                'variance': np.var(background_nonzero),
                'std': np.std(background_nonzero),
                'min': np.min(background_nonzero),
                'max': np.max(background_nonzero),
                'median': np.median(background_nonzero),
                'pixel_count': background_nonzero.size
            }
        else:
            background_stats = {'mean': 0, 'variance': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0, 'pixel_count': 0}
        
        return object_stats, background_stats
    
    def on_histogram_mode_selected(self, event=None):
        selected_mode = self.histogram_mode_var.get()
        self.update_info(f"Выбран режим: {selected_mode}")
        
        if selected_mode != "Выберите режим":
            self.apply_histogram_button.config(state=tk.NORMAL)
        else:
            self.apply_histogram_button.config(state=tk.DISABLED)
    
    def apply_histogram_mode(self):
        selected_mode = self.histogram_mode_var.get()
        
        if selected_mode == "Выберите режим":
            messagebox.showwarning("Предупреждение", "Пожалуйста, выберите режим гистограммы")
            return
        
        if self.image is None:
            messagebox.showwarning("Предупреждение", "Нет изображения для анализа")
            return
        
        try:
            if selected_mode == "Гистограмма":
                self.capture_histogram_standard()
            elif selected_mode == "Гистограмма фон+объект":
                self.capture_background_object_histogram()
            elif selected_mode == "Гистограмма до/после":
                self.capture_before_after_histogram()
            elif selected_mode == "Гистограмма фон+объект до/после":
                self.capture_before_after_strobe_histogram()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось применить режим '{selected_mode}': {str(e)}")
    
    def capture_histogram_standard(self):
        try:
            if self.image is None:
                return
            
            hist_gray, _ = self.calculate_histogram(self.image)
            
            self.captured_histogram = hist_gray
            self.captured_frame = self.image.copy()
            self.histogram_captured = True
            
            self.display_captured_histogram()
            
            self.before_histogram_data = hist_gray
            
            if self.video_capture:
                current_frame = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                current_time = current_frame / self.video_capture.get(cv2.CAP_PROP_FPS)
                self.update_info(f"Гистограмма захвачена для кадра {int(current_frame)} (время: {current_time:.2f}с)")
            else:
                self.update_info("Гистограмма захвачена для текущего изображения")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось захватить гистограмму: {str(e)}")
    
    def capture_background_object_histogram(self):
        try:
            if self.image is None:
                messagebox.showwarning("Предупреждение", "Нет изображения для анализа")
                return
            
            if self.strobe_rect is None:
                self.calculate_strobe_rect()
            
            hist_inner, hist_background = self.calculate_strobe_histograms(self.image)
            
            if hist_inner is None or hist_background is None:
                messagebox.showwarning("Предупреждение", "Не удалось выделить области")
                return
            
            self.captured_histogram = (hist_inner, hist_background)
            self.captured_frame = self.image.copy()
            self.histogram_captured = True
            
            self.before_strobe_histograms = (hist_inner, hist_background)
            
            self.display_dual_histogram(hist_inner, hist_background, title="Гистограмма: Фон vs Объект")
            
            inner_size = "N/A"
            if self.strobe_rect:
                inner_width = self.strobe_rect[2] - self.strobe_rect[0]
                inner_height = self.strobe_rect[3] - self.strobe_rect[1]
                inner_size = f"{inner_width}x{inner_height}"
            
            self.update_info(f"Гистограмма фон+объект: Область {inner_size} пикс.")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось захватить гистограмму фон+объект: {str(e)}")
    
    def capture_before_after_histogram(self):
        try:
            if self.image is None:
                messagebox.showwarning("Предупреждение", "Нет изображения для анализа")
                return
            
            hist_before, gray_before = self.calculate_histogram(self.image)
            self.before_histogram_data = hist_before
            
            if self.apply_enhancement and self.enhancement_method:
                enhanced_frame = self.apply_enhancement_to_frame(self.image)
                if len(enhanced_frame.shape) == 3 and enhanced_frame.shape[2] == 3:
                    enhanced_gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
                else:
                    enhanced_gray = enhanced_frame if len(enhanced_frame.shape) == 2 else enhanced_frame
                
                hist_after, _ = self.calculate_histogram(enhanced_gray)
                self.after_histogram_data = hist_after
            else:
                hist_after = hist_before.copy()
                self.after_histogram_data = hist_after
            
            self.display_before_after_histogram(hist_before, hist_after)
            
            self.captured_histogram = (hist_before, hist_after)
            self.captured_frame = self.image.copy()
            self.histogram_captured = True
            
            method_name = self.enhancement_method if self.apply_enhancement else "без улучшения"
            self.update_info(f"Гистограмма до/после: метод '{method_name}'")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось захватить гистограмму до/после: {str(e)}")
    
    def capture_before_after_strobe_histogram(self):
        try:
            if self.image is None:
                messagebox.showwarning("Предупреждение", "Нет изображения для анализа")
                return
            
            if self.strobe_rect is None:
                self.calculate_strobe_rect()
            
            hist_inner_before, hist_background_before = self.calculate_strobe_histograms(self.image)
            
            if hist_inner_before is None or hist_background_before is None:
                messagebox.showwarning("Предупреждение", "Не удалось выделить области")
                return
            
            self.before_strobe_histograms = (hist_inner_before, hist_background_before)
            
            if self.apply_enhancement and self.enhancement_method:
                enhanced_frame = self.apply_enhancement_to_frame(self.image)
                hist_inner_after, hist_background_after = self.calculate_strobe_histograms(enhanced_frame)
                self.after_strobe_histograms = (hist_inner_after, hist_background_after)
            else:
                hist_inner_after = hist_inner_before.copy()
                hist_background_after = hist_background_before.copy()
                self.after_strobe_histograms = (hist_inner_after, hist_background_after)
            
            self.display_before_after_strobe_histogram(
                hist_inner_before, hist_background_before,
                hist_inner_after, hist_background_after
            )
            
            self.captured_histogram = ((hist_inner_before, hist_background_before), (hist_inner_after, hist_background_after))
            self.captured_frame = self.image.copy()
            self.histogram_captured = True
            
            method_name = self.enhancement_method if self.apply_enhancement else "без улучшения"
            inner_size = "N/A"
            if self.strobe_rect:
                inner_width = self.strobe_rect[2] - self.strobe_rect[0]
                inner_height = self.strobe_rect[3] - self.strobe_rect[1]
                inner_size = f"{inner_width}x{inner_height}"
            
            self.update_info(f"Гистограмма фон+объект до/после: {inner_size} пикс., метод '{method_name}'")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось захватить гистограмму фон+объект до/после: {str(e)}")
    
    def display_dual_histogram(self, hist_inner, hist_background, title="Гистограмма: Фон vs Объект"):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
        
        x = np.linspace(0, 255, len(hist_inner))
        
        ax.plot(x, hist_inner, color='green', alpha=0.7, label='Объект (внутри строба)', linewidth=2)
        ax.plot(x, hist_background, color='red', alpha=0.7, label='Фон (вне строба)', linewidth=2, linestyle='--')
        
        ax.set_title(title)
        ax.set_xlabel('Значение пикселя')
        ax.set_ylabel('Нормализованная частота')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 255])
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        buffer.seek(0)
        hist_plot = Image.open(buffer)
        hist_photo = ImageTk.PhotoImage(hist_plot)
        
        if self.current_hist_photo:
            self.current_hist_photo = None
        
        self.hist_label.configure(image=hist_photo, text="")
        self.hist_label.image = hist_photo
        self.current_hist_photo = hist_photo
    
    def display_before_after_histogram(self, hist_before, hist_after):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6), dpi=80)
        
        x = np.linspace(0, 255, len(hist_before))
        
        ax1.plot(x, hist_before, color='blue', alpha=0.7, label='До улучшения', linewidth=2)
        ax1.set_title('Гистограмма: До улучшения')
        ax1.set_ylabel('Нормализованная частота')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 255])
        
        method_name = self.enhancement_method if self.apply_enhancement else "без улучшения"
        ax2.plot(x, hist_after, color='orange', alpha=0.7, label=f'После ({method_name})', linewidth=2)
        ax2.set_title(f'Гистограмма: После улучшения ({method_name})')
        ax2.set_xlabel('Значение пикселя')
        ax2.set_ylabel('Нормализованная частота')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 255])
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        buffer.seek(0)
        hist_plot = Image.open(buffer)
        hist_photo = ImageTk.PhotoImage(hist_plot)
        
        if self.current_hist_photo:
            self.current_hist_photo = None
        
        self.hist_label.configure(image=hist_photo, text="")
        self.hist_label.image = hist_photo
        self.current_hist_photo = hist_photo
    
    def display_before_after_strobe_histogram(self, hist_inner_before, hist_background_before, hist_inner_after, hist_background_after):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6), dpi=80)
        
        x = np.linspace(0, 255, len(hist_inner_before))
        method_name = self.enhancement_method if self.apply_enhancement else "без улучшения"
        
        ax1.plot(x, hist_inner_before, color='green', alpha=0.7, label='Объект (до)', linewidth=2)
        ax1.plot(x, hist_background_before, color='red', alpha=0.7, label='Фон (до)', linewidth=2, linestyle='--')
        ax1.set_title('До улучшения: Фон vs Объект')
        ax1.set_ylabel('Нормализованная частота')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 255])
        
        ax2.plot(x, hist_inner_after, color='green', alpha=0.7, label='Объект (после)', linewidth=2)
        ax2.plot(x, hist_background_after, color='red', alpha=0.7, label='Фон (после)', linewidth=2, linestyle='--')
        ax2.set_title(f'После улучшения ({method_name}): Фон vs Объект')
        ax2.set_xlabel('Значение пикселя')
        ax2.set_ylabel('Нормализованная частота')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 255])
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        buffer.seek(0)
        hist_plot = Image.open(buffer)
        hist_photo = ImageTk.PhotoImage(hist_plot)
        
        if self.current_hist_photo:
            self.current_hist_photo = None
        
        self.hist_label.configure(image=hist_photo, text="")
        self.hist_label.image = hist_photo
        self.current_hist_photo = hist_photo
    
    def display_captured_histogram(self):
        if self.captured_histogram is not None and self.histogram_captured:
            selected_mode = self.histogram_mode_var.get()
            
            if selected_mode == "Гистограмма":
                hist_plot = self.create_histogram_plot(self.captured_histogram)
                hist_photo = ImageTk.PhotoImage(hist_plot)
                
                if self.current_hist_photo:
                    self.current_hist_photo = None
                
                self.hist_label.configure(image=hist_photo, text="")
                self.hist_label.image = hist_photo
                self.current_hist_photo = hist_photo
                
            elif selected_mode == "Гистограмма фон+объект":
                hist_inner, hist_background = self.captured_histogram
                self.display_dual_histogram(hist_inner, hist_background, title="Гистограмма: Фон vs Объект")
                
            elif selected_mode == "Гистограмма до/после":
                hist_before, hist_after = self.captured_histogram
                self.display_before_after_histogram(hist_before, hist_after)
                
            elif selected_mode == "Гистограмма фон+объект до/после":
                (hist_inner_before, hist_background_before), (hist_inner_after, hist_background_after) = self.captured_histogram
                self.display_before_after_strobe_histogram(hist_inner_before, hist_background_before, hist_inner_after, hist_background_after)
    
    def create_histogram_plot(self, hist_gray):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
        
        x = np.linspace(0, 255, len(hist_gray))
        
        ax.plot(x, hist_gray, color='blue', alpha=0.7, label='Gray', linewidth=1)
        ax.set_title('Гистограмма')
        ax.set_xlabel('Значение пикселя')
        ax.set_ylabel('Нормализованная частота')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 255])
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        buffer.seek(0)
        return Image.open(buffer)
    
    def setup_progress_bar_and_navigation(self):
        self.progress_nav_frame = tk.Frame(self.root)
        self.progress_nav_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_top_frame = tk.Frame(self.progress_nav_frame)
        self.progress_top_frame.pack(fill=tk.X)
        
        self.progress_label = tk.Label(self.progress_top_frame, text="Прогресс:")
        self.progress_label.pack(side=tk.LEFT)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = tk.Scale(self.progress_top_frame, variable=self.progress_var, 
                                    from_=0, to=100, orient=tk.HORIZONTAL, 
                                    length=400, showvalue=True, state=tk.DISABLED,
                                    resolution=0.5,
                                    sliderlength=15)
        
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        self.progress_bar.bind("<ButtonPress-1>", self.on_seek_start)
        self.progress_bar.bind("<ButtonRelease-1>", self.on_seek_end)
        
        self.progress_var.trace_add("write", self.on_progress_change)
        
        self.nav_frame = tk.Frame(self.progress_nav_frame)
        self.nav_frame.pack(pady=(5, 0))
        
        self.nav_button_minus5 = tk.Button(self.nav_frame, text="-5", command=lambda: self.seek_frame(-6), width=3, state=tk.DISABLED)
        self.nav_button_minus5.pack(side=tk.LEFT, padx=2)
        
        self.nav_button_minus1 = tk.Button(self.nav_frame, text="-1", command=lambda: self.seek_frame(-2), width=3, state=tk.DISABLED)
        self.nav_button_minus1.pack(side=tk.LEFT, padx=2)
        
        self.pause_button = tk.Button(self.nav_frame, text="Пауза/Продолжить", command=self.toggle_pause, width=15, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.nav_button_plus1 = tk.Button(self.nav_frame, text="+1", command=lambda: self.seek_frame(0), width=3, state=tk.DISABLED)
        self.nav_button_plus1.pack(side=tk.LEFT, padx=2)
        
        self.nav_button_plus5 = tk.Button(self.nav_frame, text="+5", command=lambda: self.seek_frame(4), width=3, state=tk.DISABLED)
        self.nav_button_plus5.pack(side=tk.LEFT, padx=2)
        
        self.nav_buttons = [
            self.nav_button_minus5,
            self.nav_button_minus1,
            self.pause_button,
            self.nav_button_plus1,
            self.nav_button_plus5
        ]
    
    def seek_frame(self, frame_offset):
        if not self.video_capture or not self.is_playing:
            return
        
        try:
            current_frame = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            
            target_frame = current_frame + frame_offset
            
            if target_frame < 0:
                target_frame = 0
            elif target_frame >= total_frames:
                target_frame = total_frames - 1
            
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            ret, frame = self.video_capture.read()
            if ret:
                self.image = frame
                self.current_frame_stats = self.calculate_image_stats(frame)
                self.update_stats_display()
                self.update_both_images()
                self.histogram_captured = False
                self.update_histogram_display()
                
                progress = (target_frame / total_frames) * 100
                self.progress_var.set(progress)
                
                current_time = target_frame / self.video_capture.get(cv2.CAP_PROP_FPS)
                total_time = total_frames / self.video_capture.get(cv2.CAP_PROP_FPS)
                
                direction = "вперед" if frame_offset > 0 else "назад"
                frames_text = f"{abs(frame_offset)} кадр"
                if abs(frame_offset) > 1:
                    frames_text += "а" if abs(frame_offset) < 5 else "ов"
                
                self.update_info(f"Перемещение {direction} на {frames_text} | Кадр: {int(target_frame)} | Время: {current_time:.1f}с / {total_time:.1f}с")
                
                if not self.video_paused:
                    self.video_paused = True
                    self.update_info(f"Воспроизведение приостановлено | {self.info_label.cget('text')}")
            
        except Exception as e:
            print(f"Ошибка перемещения по кадрам: {e}")
            messagebox.showerror("Ошибка", f"Не удалось переместиться по кадрам: {str(e)}")
    
    def calculate_image_stats(self, image):
        if image is None:
            return {
                'mean': 0,
                'variance': 0,
                'std': 0,
                'frame_number': 0,
                'width': 0,
                'height': 0
            }
        
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        mean_val = np.mean(gray_image)
        variance_val = np.var(gray_image)
        std_val = np.std(gray_image)
        
        frame_number = 0
        if self.video_capture and self.is_playing:
            try:
                frame_number = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            except:
                frame_number = 0
        
        return {
            'mean': mean_val,
            'variance': variance_val,
            'std': std_val,
            'frame_number': frame_number,
            'width': image.shape[1],
            'height': image.shape[0]
        }
    
    def update_stats_display(self):
        stats = self.current_frame_stats
        
        strobe_stats_text = ""
        if self.show_strobe and self.strobe_rect is not None and self.image is not None:
            object_stats, background_stats = self.calculate_strobe_stats(self.image)
            
            if object_stats and background_stats:
                strobe_stats_text = (f" | ОБЪЕКТ: Среднее={object_stats['mean']:.1f} Дисперсия={object_stats['variance']:.1f} "
                                f"STD={object_stats['std']:.1f} пикс={object_stats['pixel_count']} | "
                                f"ФОН: Среднее={background_stats['mean']:.1f} Дисперсия={background_stats['variance']:.1f} "
                                f"STD={background_stats['std']:.1f} пикс={background_stats['pixel_count']}")
        
        stats_text = (f"Общая статистика: Среднее={stats['mean']:.1f} | "
                    f"Дисперсия={stats['variance']:.1f} | "
                    f"STD={stats['std']:.1f} | "
                    f"Кадр: {stats['frame_number']} | "
                    f"Размер: {stats['width']}x{stats['height']}"
                    f"{strobe_stats_text}")
        
        try:
            if self.stats_label.winfo_exists():
                self.stats_label.config(text=stats_text)
        except tk.TclError:
            pass
        
        self.update_levels_stats_display()
    
    def on_seek_start(self, event):
        self.seeking = True
        self.progress_update_paused = True
        self._seek_start_time = time.time()
        self._fast_seek_counter = 0
        
        if self.is_playing and not self.video_paused:
            self.video_paused = True
    
    def on_seek_end(self, event):
        self.seeking = False
        self.progress_update_paused = False
        
        self.perform_seek(self.progress_var.get())
        
        if self.is_playing:
            self.video_paused = True
    
    def on_progress_change(self, *args):
        if not self.seeking or not self.video_capture:
            return
        
        current_time = time.time()
        if current_time - self._last_seek_time < 0.03:
            return
        
        self._last_seek_time = current_time
    
        if current_time - self._seek_start_time < 1.0:
            self._fast_seek_counter += 1
            if self._fast_seek_counter % 2 == 0:
                return
                
        self.perform_seek(self.progress_var.get())
    
    def perform_seek(self, progress):
        if not self.video_capture or not self.is_playing:
            return
        
        try:
            total_frames = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_frames <= 0:
                return
            
            target_frame = int((progress / 100) * total_frames)
            
            if abs(target_frame - self._last_seek_frame) < 5:
                return
            
            self._last_seek_frame = target_frame
            
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            ret, frame = self.video_capture.read()
            if ret:
                self.image = frame
                self.current_frame_stats = self.calculate_image_stats(frame)
                self.update_stats_display()
                self.update_both_images()
                self.histogram_captured = False
                self.update_histogram_display()
                
                current_time = target_frame / self.video_capture.get(cv2.CAP_PROP_FPS)
                total_time = total_frames / self.video_capture.get(cv2.CAP_PROP_FPS)
                self.update_info(f"Перемотка: {current_time:.1f}с / {total_time:.1f}с")
                
        except Exception as e:
            print(f"Ошибка перемотки: {e}")
    
    def enable_video_controls(self, enable=True):
        try:
            state = tk.NORMAL if enable else tk.DISABLED
            if self.pause_button.winfo_exists():
                self.pause_button.config(state=state)
            if self.stop_button.winfo_exists():
                self.stop_button.config(state=state)
            if self.histogram_mode_combo.winfo_exists():
                self.histogram_mode_combo.config(state=state if enable else "disabled")
            if self.apply_histogram_button.winfo_exists():
                self.apply_histogram_button.config(state=state if enable else tk.DISABLED)
            if self.strobe_button.winfo_exists():
                self.strobe_button.config(state=state)
            if self.enhancement_mode_combo.winfo_exists():
                self.enhancement_mode_combo.config(state=state if enable else "disabled")
            
            for button in self.nav_buttons:
                if button.winfo_exists():
                    button.config(state=state)
            
            if enable and self.progress_nav_frame.winfo_exists():
                self.progress_nav_frame.pack(fill=tk.X, padx=10, pady=5)
                self.progress_bar.config(state=tk.NORMAL)
            elif self.progress_nav_frame.winfo_exists():
                self.progress_nav_frame.pack_forget()
                self.progress_bar.config(state=tk.DISABLED)
        except tk.TclError:
            pass
    
    def capture_histogram(self):
        self.capture_histogram_standard()
    
    def update_histogram_display(self):
        if not self.histogram_captured:
            if self.current_hist_photo:
                self.current_hist_photo = None
            self.hist_label.configure(image='', text="Гистограмма не захвачена")
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        if file_path:
            self.stop_video()
            self.image = cv2.imread(file_path)
            if self.image is not None:
                self.current_frame_stats = self.calculate_image_stats(self.image)
                self.update_stats_display()
                
                self.update_both_images()
                self.update_info(f"Изображение: {os.path.basename(file_path)} | Размер: {self.image.shape[1]}x{self.image.shape[0]}")
                self.enable_video_controls(False)
                
                if self.histogram_mode_combo.winfo_exists():
                    self.histogram_mode_combo.config(state="readonly")
                if self.apply_histogram_button.winfo_exists():
                    self.apply_histogram_button.config(state=tk.NORMAL)
                if self.strobe_button.winfo_exists():
                    self.strobe_button.config(state=tk.NORMAL)
                if self.enhancement_mode_combo.winfo_exists():
                    self.enhancement_mode_combo.config(state="readonly")
                
                for button in self.nav_buttons:
                    if button.winfo_exists():
                        button.config(state=tk.DISABLED)
                
                self.histogram_mode_var.set("Гистограмма")
                self.on_histogram_mode_selected()
                self.apply_histogram_mode()
                
                if self.enhancement_method and self.apply_enhancement:
                    self.update_both_images()
            else:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображение")
    
    def load_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        )
        if file_path:
            self.stop_video()
            self.video_capture = cv2.VideoCapture(file_path)
            if self.video_capture.isOpened():
                self.is_playing = True
                self.video_paused = False
                self.seeking = False
                self.progress_update_paused = False
                
                self._last_seek_frame = 0
                self._last_seek_time = 0
                self.histogram_captured = False
                self.update_histogram_display()
                
                self.frame_queue.clear()
                self.enhanced_frame_queue.clear()
                
                fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                
                self.update_info(f"Видео: {os.path.basename(file_path)} | FPS: {fps:.1f} | Кадры: {total_frames} | Время: {duration:.1f}с")
                self.enable_video_controls(True)
                
                self.video_thread = threading.Thread(target=self.video_reader_thread, daemon=True)
                self.video_thread.start()
                
                self.update_video_display()
            else:
                messagebox.showerror("Ошибка", "Не удалось загрузить видео")
    
    def video_reader_thread(self):
        while self.is_playing and self.video_capture and self.video_capture.isOpened():
            try:
                if self.video_paused or self.seeking:
                    time.sleep(0.01)
                    continue
                
                current_time = time.time()
                if current_time - self.last_frame_time >= self.frame_interval:
                    ret, frame = self.video_capture.read()
                    if ret:
                        if len(self.frame_queue) < self.frame_queue.maxlen:
                            self.frame_queue.append(frame)
                        
                        if self.apply_enhancement and self.enhancement_method:
                            enhanced_frame = self.apply_enhancement_to_frame(frame)
                            if len(self.enhanced_frame_queue) < self.enhanced_frame_queue.maxlen:
                                self.enhanced_frame_queue.append(enhanced_frame)
                        
                        self.last_frame_time = current_time
                    else:
                        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"Ошибка в потоке чтения видео: {e}")
                break
    
    def update_video_display(self):
        if not self.is_playing or not self.video_capture:
            return
            
        try:
            if not self.video_paused and not self.seeking:
                if self.frame_queue:
                    original_frame = self.frame_queue.popleft() if self.frame_queue else None
                    
                    if original_frame is not None:
                        self.image = original_frame
                        self.current_frame_stats = self.calculate_image_stats(original_frame)
                        self.update_stats_display()
                        self.update_both_images()
                
                if self.apply_enhancement and self.enhancement_method and self.enhanced_frame_queue:
                    pass
            
            if (self.video_capture and not self.seeking and 
                not self.progress_update_paused and not self.video_paused):
                
                current_frame = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                total_frames = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
                
                if total_frames > 0:
                    progress = (current_frame / total_frames) * 100
                    
                    self.progress_var.trace_remove("write", self.progress_var.trace_info()[0][1])
                    self.progress_bar.set(progress)
                    self.progress_var.trace_add("write", self.on_progress_change)
        
            if self.is_playing:
                self.progress_updater = self.root.after(33, self.update_video_display)
                
        except Exception as e:
            print(f"Ошибка обновления отображения: {e}")
            self.stop_video()

    def update_both_images(self):
        if self.image is None:
            return
        
        try:
            display_original = self.image.copy()
            
            if len(display_original.shape) == 3:
                if display_original.shape[2] == 3:
                    gray_original = cv2.cvtColor(display_original, cv2.COLOR_BGR2GRAY)
                elif display_original.shape[2] == 4:
                    gray_original = cv2.cvtColor(display_original, cv2.COLOR_BGRA2GRAY)
                else:
                    gray_original = display_original[:, :, 0]
            else:
                gray_original = display_original
            
            if self.show_strobe and self.strobe_rect is not None:
                if len(gray_original.shape) == 2:
                    image_with_strobe = cv2.cvtColor(gray_original, cv2.COLOR_GRAY2BGR)
                else:
                    image_with_strobe = gray_original.copy()
                
                image_with_strobe = self.apply_strobe_to_frame(image_with_strobe)
                gray_original = cv2.cvtColor(image_with_strobe, cv2.COLOR_BGR2RGB)
            
            original_display = self.prepare_display_image(gray_original)
            self.update_image_display(original_display, is_original=True)
            
            if self.apply_enhancement and self.enhancement_method:
                enhanced_frame = self.apply_enhancement_to_frame(self.image)
                
                if len(enhanced_frame.shape) == 3:
                    if enhanced_frame.shape[2] == 3:
                        enhanced_gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
                    elif enhanced_frame.shape[2] == 4:
                        enhanced_gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGRA2GRAY)
                    else:
                        enhanced_gray = enhanced_frame[:, :, 0]
                else:
                    enhanced_gray = enhanced_frame
                
                if self.show_strobe and self.strobe_rect is not None:
                    if len(enhanced_gray.shape) == 2:
                        enhanced_with_strobe = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
                    else:
                        enhanced_with_strobe = enhanced_gray.copy()
                    
                    enhanced_with_strobe = self.apply_strobe_to_frame(enhanced_with_strobe)
                    enhanced_gray = cv2.cvtColor(enhanced_with_strobe, cv2.COLOR_BGR2RGB)
                
                enhanced_display = self.prepare_display_image(enhanced_gray)
                self.update_image_display(enhanced_display, is_original=False)
            else:
                enhanced_display = self.prepare_display_image(gray_original)
                self.update_image_display(enhanced_display, is_original=False)
            
            self.update_stats_display()
            
        except Exception as e:
            print(f"Ошибка обновления изображений: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_image_display(self, image_photo, is_original=True):
        if is_original:
            if self.current_photo:
                self.current_photo = None
            
            self.image_label.configure(image=image_photo, text="")
            self.image_label.image = image_photo
            self.current_photo = image_photo
        else:
            if self.current_enhanced_photo:
                self.current_enhanced_photo = None
            
            self.enhanced_label.configure(image=image_photo, text="")
            self.enhanced_label.image = image_photo
            self.current_enhanced_photo = image_photo
    
    def toggle_pause(self):
        if self.video_capture:
            self.video_paused = not self.video_paused
        
            status = "приостановлено" if self.video_paused else "продолжено"
            self.update_info(f"Воспроизведение {status}")
        
            if not self.video_paused:
                self.frame_queue.clear()
                self.enhanced_frame_queue.clear()
                self.progress_update_paused = False
    
    def stop_video(self):
        self.is_playing = False
        self.video_paused = False
        self.seeking = False
        self.progress_update_paused = False
        
        if self.progress_updater:
            self.root.after_cancel(self.progress_updater)
            self.progress_updater = None
            
        if self.video_capture:
            try:
                self.video_capture.release()
            except Exception as e:
                print(f"Ошибка освобождения видео: {e}")
            finally:
                self.video_capture = None
                
        self.frame_queue.clear()
        self.enhanced_frame_queue.clear()
        
        try:
            self.enable_video_controls(False)
            self.update_info("Воспроизведение остановлено")
        except tk.TclError:
            pass
    
    def fast_histogram(self, gray_image, bins=32):
        step = 256 // bins
        indices = gray_image // step
        hist = np.bincount(indices.ravel(), minlength=bins)
        return hist.astype(np.float32)
    
    def calculate_histogram(self, image):
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                gray_image = image[:, :, 0]
        else:
            gray_image = image
        
        hist_gray = self.fast_histogram(gray_image, bins=128)
        
        if hist_gray.max() > 0:
            hist_gray = hist_gray / hist_gray.max()
        
        return hist_gray, gray_image
    
    def prepare_display_image(self, gray_image):
        h, w = gray_image.shape[:2]
        max_width, max_height = 400, 300
        
        scale = min(max_width/w, max_height/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized_image = cv2.resize(gray_image, (new_w, new_h))
        
        image_pil = Image.fromarray(resized_image)
        photo = ImageTk.PhotoImage(image_pil)
        
        return photo
    
    def update_info(self, text):
        try:
            self.info_label.config(text=text)
        except tk.TclError:
            pass
    
    def cleanup(self):
        self.stop_video()
        
        if self.current_photo:
            self.current_photo = None
        if self.current_enhanced_photo:
            self.current_enhanced_photo = None
        if self.current_hist_photo:
            self.current_hist_photo = None
        
        try:
            if self.image_label.winfo_exists():
                self.image_label.configure(image='', text="Исходное видео")
            if self.enhanced_label.winfo_exists():
                self.enhanced_label.configure(image='', text="Улучшенное видео")
            if self.hist_label.winfo_exists():
                self.hist_label.configure(image='', text="Гистограмма не захвачена")
            if self.stats_label.winfo_exists():
                self.stats_label.config(text="Статистика: Среднее: -- | Дисперсия: -- | STD: -- | Кадр: -- | Размер: --x--")
            if self.levels_stats_label.winfo_exists():
                self.levels_stats_label.config(text="Параметры Levels: Peak: -- | Black: -- | White: -- | Gamma: --")
        except tk.TclError:
            pass

def main():
    root = tk.Tk()
    app = HistogramAnalyzer(root)
    
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()