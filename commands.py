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

class RecognizeTimeCommand(Command):
    """
    Пошаговое распознавание времени на часах с визуализацией всех промежуточных шагов.
    Каждый шаг можно запускать кнопкой "Следующий шаг".
    """
    def __init__(self, editor):
        super().__init__(editor)
        self.steps = [
            self.find_clock_face,
            self.detect_edges,
            self.detect_all_lines,
            self.select_hands,
            self.compute_angles,
            self.display_time
        ]
        self.current_step = 0
        self.result = None

    def start(self):
        """Запуск пошагового распознавания."""
        img_np = np.array(self.editor.displayed_img)
        self.gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
        self.result = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        self.next_step()

    def next_step(self):
        """Выполнить следующий шаг."""
        if self.current_step >= len(self.steps):
            messagebox.showinfo("Готово", "Все шаги выполнены")
            return

        step_func = self.steps[self.current_step]
        step_func()
        self.current_step += 1

    def show_intermediate(self, img, title=None):
        """Показать промежуточное изображение в редакторе."""
        self.editor.displayed_img = Image.fromarray(img)
        self.editor.show_image(self.editor.displayed_img)
        if title:
            print(title)

    # --- Шаги ---
    def find_clock_face(self):
        circles = cv2.HoughCircles(
            self.gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=self.gray.shape[0]//2,
            param1=50, param2=30,
            minRadius=self.gray.shape[0]//4, maxRadius=self.gray.shape[0]//2
        )
        if circles is None:
            messagebox.showinfo("Результат", "Циферблат не найден")
            self.show_intermediate(self.result)
            return

        self.cx, self.cy, self.r = np.uint16(np.around(circles[0,0]))
        circ_img = self.result.copy()
        cv2.circle(circ_img, (self.cx, self.cy), self.r, (0,255,0), 2)
        self.show_intermediate(circ_img, "Найден циферблат")

    def detect_edges(self):
        edges = cv2.Canny(self.gray, 50, 150)
        self.edges = edges
        self.show_intermediate(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "Границы (Canny)")

    def detect_all_lines(self):
        lines = cv2.HoughLinesP(
            self.edges, 1, np.pi/180, threshold=30,
            minLineLength=self.r//4, maxLineGap=10
        )
        self.lines = lines
        all_lines_img = self.result.copy()
        if lines is not None:
            for x1, y1, x2, y2 in lines[:,0]:
                cv2.line(all_lines_img, (x1, y1), (x2, y2), (0, 255, 255), 3)  # толщина = 3
        self.show_intermediate(all_lines_img, "Все найденные линии")


    def select_hands(self):
        if self.lines is None:
            messagebox.showinfo("Результат", "Линии не найдены")
            return

        arrow_lines = []
        for x1, y1, x2, y2 in self.lines[:,0]:
            dist1 = np.hypot(x1 - self.cx, y1 - self.cy)
            dist2 = np.hypot(x2 - self.cx, y2 - self.cy)
            if dist1 < self.r*0.3 or dist2 < self.r*0.3:  # чуть больше допуск
                arrow_lines.append((x1, y1, x2, y2, np.hypot(x2-x1, y2-y1)))

        if len(arrow_lines) < 2:
            messagebox.showinfo("Результат", "Недостаточно линий для определения стрелок")
            return

        # сортировка по длине
        arrow_lines.sort(key=lambda x: x[4], reverse=True)
        self.minute_line = arrow_lines[0]
        self.hour_line = arrow_lines[1]

        hands_img = self.result.copy()
        for line, color in zip([self.hour_line, self.minute_line], [(255,0,0),(0,0,255)]):
            x1, y1, x2, y2, _ = line
            cv2.line(hands_img, (x1, y1), (x2, y2), color, 3)  # толщина = 3
        self.show_intermediate(hands_img, "Отобранные стрелки")


    def compute_angles(self):
        def tip(x1, y1, x2, y2):
            """Возвращает координаты кончика (точка, более далёкая от центра циферблата)."""
            d1 = np.hypot(x1 - self.cx, y1 - self.cy)
            d2 = np.hypot(x2 - self.cx, y2 - self.cy)
            return (x2, y2) if d2 > d1 else (x1, y1)

        def angle_from_center(line):
            """
            Возвращает угол в градусах:
            0° = вверх (12 часов), по часовой стрелке: право=90°, низ=180°, лево=270°.
            """
            x1, y1, x2, y2, *_ = line
            x_tip, y_tip = tip(x1, y1, x2, y2)
            dx = x_tip - self.cx
            dy = self.cy - y_tip  # инвертируем ось Y (в изображениях Y растёт вниз)
            return (np.degrees(np.arctan2(dx, dy)) % 360)

        def length(line):
            x1, y1, x2, y2, *_ = line
            return np.hypot(x2 - x1, y2 - y1)

        # --- 1) Получаем углы двух линий ---
        a1 = angle_from_center(self.minute_line)
        a2 = angle_from_center(self.hour_line)

        # --- 2) Подстраховка: определим какая стрелка минутная/часовая ---
        # Обычно минутная длиннее. Если у вас уже гарантировано разделение
        # на self.minute_line / self.hour_line — этот блок всё равно не повредит.
        len1 = length(self.minute_line)
        len2 = length(self.hour_line)

        if len2 > len1:
            # Похоже, self.hour_line на самом деле длиннее — вероятно это минутная.
            minute_angle, hour_angle = a2, a1
        else:
            minute_angle, hour_angle = a1, a2

        # --- 3) Вычисляем минуты из минутной стрелки ---
        # Минуты кратны 6°. Округляем к ближайшей минуте.
        minute = int(round(minute_angle / 6.0)) % 60

        # Если из-за округления получили 60, нормализуем к 0 и потом прибавим к часу.
        minute_carry = 1 if minute == 0 and (minute_angle % 360) > 354 else 0  # редкий случай близко к 360°

        # --- 4) Часы из формулы: hour_angle ≈ 30·H + 0.5·M ---
        # Используем уже округлённые минуты, чтобы согласовать обе стрелки.
        # Получим целые часы, затем нормализуем в 0..11.
        hour_raw = (hour_angle - 0.5 * minute) / 30.0
        hour = int(round(hour_raw)) % 12

        # Перенос часа, если минутная «перекатилась» через 12 при округлении.
        hour = (hour + minute_carry) % 12

        # --- 5) Сохраняем всё в объект ---
        self.minute_angle = minute_angle
        self.hour_angle = hour_angle
        self.minute = minute
        self.hour = hour

        # --- 6) Сообщение пользователю ---
        msg = (
            f"Часовая стрелка:\n"
            f"  Угол = {self.hour_angle:.1f}°\n"
            f"  Часы = {self.hour}\n\n"
            f"Минутная стрелка:\n"
            f"  Угол = {self.minute_angle:.1f}°\n"
            f"  Минуты = {self.minute:02d}"
        )
        messagebox.showinfo("Вычисленные углы и время", msg)
        print(msg)
        self.show_intermediate(self.result, "Вычислены углы стрелок")



    def display_time(self):
        cv2.putText(self.result, f"{int(self.hour):02d}:{self.minute:02d}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        self.show_intermediate(self.result, "Время на часах")
        messagebox.showinfo("Распознанное время", f"{int(self.hour):02d}:{self.minute:02d}")
