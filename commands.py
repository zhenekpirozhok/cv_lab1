from tkinter import simpledialog
import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image
import cv2

class Command:
    def __init__(self, editor):
        self.editor = editor
        self.prev_image = editor.displayed_img.copy()

    def execute(self):
        pass

    def undo(self):
        self.editor.displayed_img = self.prev_image
        self.editor.show_image(self.prev_image)

class ParamCommand(Command):
    params = {}  # {'param_name': (type, default_value, prompt)}

    def execute(self):
        kwargs = self.ask_parameters()
        self.execute_with_params(**kwargs)

    def ask_parameters(self):
        kwargs = {}
        for name, (typ, default, prompt) in self.params.items():
            root = self.editor.root
            if typ == int:
                val = simpledialog.askinteger("Параметр", prompt, initialvalue=default, parent=root)
            elif typ == float:
                val = simpledialog.askfloat("Параметр", prompt, initialvalue=default, parent=root)
            else:
                val = default
            kwargs[name] = val if val is not None else default
        return kwargs

    def execute_with_params(self, **kwargs):
        pass


class GrayscaleAverageCommand(Command):
    def execute(self):
        img_np = np.array(self.editor.displayed_img)
        if img_np.ndim == 3:
            gray = img_np.mean(axis=2).astype(np.uint8)
        else:
            gray = img_np.astype(np.uint8)
        self.editor.displayed_img = Image.fromarray(gray)
        self.editor.show_image(self.editor.displayed_img)

class GrayscaleHSVCommand(Command):
    def execute(self):
        img_np = np.array(self.editor.displayed_img)
        if img_np.ndim == 3:
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            v = hsv[:, :, 2]
        else:
            v = img_np
        self.editor.displayed_img = Image.fromarray(v)
        self.editor.show_image(self.editor.displayed_img)

class BinarizationCommand(Command):
    def execute(self):
        img_np = np.array(self.editor.displayed_img)

        if img_np.ndim == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        _, bin_img = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.editor.displayed_img = Image.fromarray(bin_img)
        self.editor.show_image(self.editor.displayed_img)

class NormalizeCommand(Command):
    def execute(self):
        img_np = np.array(self.editor.displayed_img).astype(np.float32)
        if img_np.ndim == 3:  # RGB
            for i in range(3):
                img_np[:,:,i] = cv2.normalize(img_np[:,:,i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        else:  # Ч/Б
            img_np = cv2.normalize(img_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_np = img_np.astype(np.uint8)
        self.editor.displayed_img = Image.fromarray(img_np)
        self.editor.show_image(self.editor.displayed_img)


class ContrastStretchCommand(Command):
    def execute(self):
        img_np = np.array(self.editor.displayed_img).astype(np.float32)
        if img_np.ndim == 3:  # RGB
            for i in range(3):
                min_val = np.min(img_np[:,:,i])
                max_val = np.max(img_np[:,:,i])
                if max_val > min_val:
                    img_np[:,:,i] = (img_np[:,:,i] - min_val) * 255 / (max_val - min_val)
        else:
            min_val = np.min(img_np)
            max_val = np.max(img_np)
            if max_val > min_val:
                img_np = (img_np - min_val) * 255 / (max_val - min_val)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        self.editor.displayed_img = Image.fromarray(img_np)
        self.editor.show_image(self.editor.displayed_img)


class EqualizeHistCommand(Command):
    def execute(self):
        img_np = np.array(self.editor.displayed_img)
        if img_np.ndim == 3:  # RGB
            for i in range(3):
                img_np[:,:,i] = cv2.equalizeHist(img_np[:,:,i])
        else:  # Ч/Б
            img_np = cv2.equalizeHist(img_np)
        self.editor.displayed_img = Image.fromarray(img_np)
        self.editor.show_image(self.editor.displayed_img)

class GaussianBlurCommand(ParamCommand):
    params = {
        'ksize': (int, 7, "Введите размер ядра (нечетное число):"),
        'sigma': (float, 1.5, "Введите sigma (стандартное отклонение):")
    }
    def execute_with_params(self, ksize, sigma):
        if ksize % 2 == 0:
            ksize += 1
        img_np = np.array(self.editor.displayed_img)
        if img_np.ndim == 3:
            blurred = np.zeros_like(img_np)
            for i in range(3):
                blurred[:,:,i] = cv2.GaussianBlur(img_np[:,:,i], (ksize, ksize), sigma)
        else:
            blurred = cv2.GaussianBlur(img_np, (ksize, ksize), sigma)
        self.editor.displayed_img = Image.fromarray(blurred)
        self.editor.show_image(self.editor.displayed_img)


class SharpenCommand(ParamCommand):
    params = {
        'alpha': (float, 1.5, "Введите коэффициент резкости (alpha):")
    }
    def execute_with_params(self, alpha):
        img_np = np.array(self.editor.displayed_img).astype(np.uint8)
        if img_np.ndim == 3:
            sharp = np.zeros_like(img_np)
            for i in range(3):
                lap = cv2.Laplacian(img_np[:,:,i], cv2.CV_16S)
                lap = cv2.convertScaleAbs(lap)
                sharp[:,:,i] = cv2.convertScaleAbs(img_np[:,:,i] - alpha * lap)
        else:
            lap = cv2.Laplacian(img_np, cv2.CV_16S)
            lap = cv2.convertScaleAbs(lap)
            sharp = cv2.convertScaleAbs(img_np - alpha * lap)
        self.editor.displayed_img = Image.fromarray(sharp)
        self.editor.show_image(self.editor.displayed_img)


class SobelEdgeCommand(Command):
    def execute(self):
        img_np = np.array(self.editor.displayed_img).astype(np.float32)
        if img_np.ndim == 3:
            sobel_img = np.zeros_like(img_np)
            for i in range(3):
                sobel_x = cv2.Sobel(img_np[:,:,i], cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(img_np[:,:,i], cv2.CV_64F, 0, 1, ksize=3)
                sobel = cv2.magnitude(sobel_x, sobel_y)
                sobel_img[:,:,i] = cv2.convertScaleAbs(sobel)
        else:
            sobel_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobel_x, sobel_y)
            sobel_img = cv2.convertScaleAbs(sobel)
        self.editor.displayed_img = Image.fromarray(sobel_img.astype(np.uint8))
        self.editor.show_image(self.editor.displayed_img)

class ShiftCommand(ParamCommand):
    params = {
        'dx': (int, 50, "Введите сдвиг по горизонтали (dx):"),
        'dy': (int, 30, "Введите сдвиг по вертикали (dy):")
    }
    def execute_with_params(self, dx, dy):
        img_np = np.array(self.editor.displayed_img)
        shifted = np.roll(img_np, shift=(dy, dx), axis=(0,1))
        self.editor.displayed_img = Image.fromarray(shifted)
        self.editor.show_image(self.editor.displayed_img)


class RotateCommand(ParamCommand):
    params = {
        'angle': (float, 45, "Введите угол поворота (градусы):"),
        'scale': (float, 1.0, "Введите масштаб (scale):")
    }
    def execute_with_params(self, angle, scale):
        img_np = np.array(self.editor.displayed_img)
        h, w = img_np.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(img_np, M, (w, h))
        self.editor.displayed_img = Image.fromarray(rotated)
        self.editor.show_image(self.editor.displayed_img)

class HoughLinesCommand(ParamCommand):
    params = {
        'rho': (float, 1, "Введите разрешение по расстоянию (rho):"),
        'theta_deg': (float, 1, "Введите разрешение по углу (градусы):"),
        'threshold': (int, 120, "Введите порог голосов:"),
        'min_line_length': (int, 50, "Введите минимальную длину линии:"),
        'max_line_gap': (int, 1, "Введите максимальный разрыв линии:")
    }

    def execute_with_params(self, rho, theta_deg, threshold, min_line_length, max_line_gap):
        img_np = np.array(self.editor.displayed_img)

        # Порог для выделения границ
        _, edges = cv2.threshold(img_np, 50, 255, cv2.THRESH_BINARY)

        # Поиск линий
        lines = cv2.HoughLinesP(
            edges,
            rho=rho,
            theta=np.deg2rad(theta_deg),
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        # Копия для рисования
        img_hough = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR) if img_np.ndim == 2 else img_np.copy()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)

        self.editor.displayed_img = Image.fromarray(img_hough)
        self.editor.show_image(self.editor.displayed_img)


class HoughCirclesCommand(ParamCommand):
    params = {
        'dp': (float, 1.0, "Введите масштаб аккумулятора (dp):"),
        'min_dist': (float, 30, "Введите минимальное расстояние между центрами:"),
        'param1': (float, 80, "Введите порог для Canny (param1):"),
        'param2': (float, 50, "Введите порог голосов (param2):"),
        'min_radius': (int, 20, "Введите минимальный радиус:"),
        'max_radius': (int, 40, "Введите максимальный радиус:")
    }

    def execute_with_params(self, dp, min_dist, param1, param2, min_radius, max_radius):
        img_np = np.array(self.editor.displayed_img)

        # Используем изображение как есть, предполагаем, что оно уже подготовлено
        img_to_use = img_np

        # Поиск окружностей
        circles = cv2.HoughCircles(
            img_to_use,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )

      
        img_hough = cv2.cvtColor(img_to_use, cv2.COLOR_GRAY2BGR) if img_to_use.ndim == 2 else img_to_use.copy()

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                center = (c[0], c[1])
                radius = c[2]
                cv2.circle(img_hough, center, radius, (0, 255, 0), 2)
                cv2.circle(img_hough, center, 2, (0, 0, 255), 3)

        self.editor.displayed_img = Image.fromarray(img_hough)
        self.editor.show_image(self.editor.displayed_img)


class LocalStatsCommand(ParamCommand):
    params = {
        'window_size': (int, 7, "Введите размер окна (нечетное число):"),
    }

    def execute_with_params(self, window_size):
        if window_size % 2 == 0:
            window_size += 1

        img_np = np.array(self.editor.displayed_img)
        if img_np.ndim == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np

        # Считаем локальную дисперсию
        mean = cv2.blur(img_gray.astype(np.float32), (window_size, window_size))
        mean_sq = cv2.blur(img_gray.astype(np.float32)**2, (window_size, window_size))
        variance = mean_sq - mean**2
        variance = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)

        self.editor.displayed_img = Image.fromarray(variance.astype(np.uint8))
        self.editor.show_image(self.editor.displayed_img)


class RegionGrowCommand(ParamCommand):
    params = {
        'tolerance': (int, 15, "Введите допуск (разброс яркости):")
    }

    def execute_with_params(self, tolerance):
        img_np = np.array(self.editor.displayed_img)
        if img_np.ndim == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        seed = {'x': None, 'y': None}
        label = self.editor.image_label

        messagebox.showinfo("Инфо", "Кликните по изображению, чтобы выбрать точку затравки.")

        def on_click(event):
            # Получаем координаты клика относительно изображения
            seed['x'], seed['y'] = event.x, event.y
            label.unbind("<Button-1>")  # отключаем обработчик после клика
            self.region_grow(gray, seed['x'], seed['y'], tolerance)

        label.bind("<Button-1>", on_click)

    def region_grow(self, gray, seed_x, seed_y, tolerance):
        h, w = gray.shape
        mask = np.zeros_like(gray, dtype=np.uint8)
        seed_value = gray[seed_y, seed_x]
        stack = [(seed_x, seed_y)]
        mask[seed_y, seed_x] = 255

        while stack:
            x, y = stack.pop()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] == 0:
                        if abs(int(gray[ny, nx]) - int(seed_value)) <= tolerance:
                            mask[ny, nx] = 255
                            stack.append((nx, ny))

        # Наложим результат на исходное изображение
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result[mask == 255] = [0, 255, 0]  # выделение области зелёным

        self.editor.displayed_img = Image.fromarray(result)
        self.editor.show_image(self.editor.displayed_img)
