import cv2
import numpy as np
import math
from PIL import Image
from tkinter import messagebox, simpledialog
from commands import Command, GaussianBlurCommand, GrayscaleHSVCommand, RotateCommand  # ✅ добавили RotateCommand

class RecognizeTimeCommand(Command):
    """
    Пошаговое распознавание времени на аналоговых часах.
    Каждый шаг можно выполнять кнопкой 'Следующий шаг' в редакторе.
    """
    def __init__(self, editor):
        super().__init__(editor)
        self.steps = [
            self.step_edges,
            self.step_contours,
            self.step_find_ellipse,
            self.step_perspective,
            self.step_detect_lines,
            self.step_calculate_time
        ]
        self.current_step = 0
        self.result = None
        self.gray = None
        self.edges = None
        self.contours = None
        self.ellipse = None
        self.filtered_lines = None

    def start(self):
        """Запуск пошагового распознавания."""
        # Выполняем уже готовые команды: Grayscale и GaussianBlur
        GrayscaleHSVCommand(self.editor).execute()
        GaussianBlurCommand(self.editor).execute()

        # Сохраняем результат
        img_np = np.array(self.editor.displayed_img)
        self.gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
        self.result = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        self.current_step = 0
        self.next_step()

    def next_step(self):
        """Выполнить следующий шаг."""
        if self.current_step >= len(self.steps):
            messagebox.showinfo("Готово", "Распознавание времени завершено.")
            return

        step_func = self.steps[self.current_step]
        step_func()
        self.current_step += 1

    def show_intermediate(self, img, title=None):
        """Показать промежуточное изображение в редакторе."""
        self.editor.displayed_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.editor.show_image(self.editor.displayed_img)
        if title:
            print(title)

    # =====================
    # ШАГИ РАСПОЗНАВАНИЯ
    # =====================

    def step_edges(self):
        """1. Выделение краёв (Canny)."""
        self.edges = cv2.Canny(self.gray, 50, 150)
        edges_bgr = cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR)
        self.show_intermediate(edges_bgr, "Edges (Canny)")

    def step_contours(self):
        """2. Поиск контуров циферблата."""
        contours, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = self.result.copy()
        cv2.drawContours(output, contours, -1, (255, 0, 0), 1)
        self.contours = contours
        self.show_intermediate(output, "Contours")

    def step_find_ellipse(self):
        """3. Нахождение лучшего эллипса (контур часов)."""
        if not self.contours:
            print("Контуры не найдены.")
            return

        height, width = self.gray.shape[:2]
        img_center = np.array([width // 2, height // 2])

        def contour_center(cnt):
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                return np.array([0, 0])
            return np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

        candidate_contours = [cnt for cnt in self.contours if 500 < cv2.contourArea(cnt) < 3000]
        if not candidate_contours:
            print("Подходящие контуры не найдены.")
            return

        best_contour = min(candidate_contours, key=lambda cnt: np.linalg.norm(contour_center(cnt) - img_center))

        if len(best_contour) >= 5:
            self.ellipse = cv2.fitEllipse(best_contour)
            output = self.result.copy()
            cv2.ellipse(output, self.ellipse, (0, 255, 0), 2)
            self.show_intermediate(output, "Fitted Ellipse")
        else:
            print("Недостаточно точек для эллипса.")

    def step_perspective(self):
        """4. Перспективное выравнивание и вращение (через RotateCommand)."""
        if self.ellipse is None:
            print("Эллипс не найден.")
            return

        (cx, cy), (major_axis, minor_axis), angle = self.ellipse
        theta = np.deg2rad(angle)

        def ellipse_point(deg):
            rad = np.deg2rad(deg)
            x = cx + (major_axis/2)*np.cos(rad)*np.cos(theta) - (minor_axis/2)*np.sin(rad)*np.sin(theta)
            y = cy + (major_axis/2)*np.cos(rad)*np.sin(theta) + (minor_axis/2)*np.sin(rad)*np.cos(theta)
            return [x, y]

        src_pts = np.array([
            ellipse_point(0),
            ellipse_point(90),
            ellipse_point(180),
            ellipse_point(270)
        ], dtype="float32")

        size = 400
        dst_pts = np.array([
            [size-1, size//2],
            [size//2, 0],
            [0, size//2],
            [size//2, size-1]
        ], dtype="float32")

        # Перспективное выравнивание
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(self.result, M, (size, size))
        warped = cv2.flip(warped, 1)

        # Отобразить результат перед вращением
        self.result = warped
        self.show_intermediate(warped, "Perspective warped")

        # ✅ Применить вращение через твою команду RotateCommand (с UI для ввода параметров)
        rotate_cmd = RotateCommand(self.editor)
        rotate_cmd.execute()  # автоматически спросит угол и масштаб

        # Обновляем результат из редактора
        self.result = np.array(self.editor.displayed_img)
        self.show_intermediate(self.result, "After RotateCommand")


    def step_detect_lines(self):
        """5. Поиск стрелок (HoughLinesP)."""
        gray = cv2.cvtColor(self.result, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=20)
        output = self.result.copy()
        center = (gray.shape[1] // 2, gray.shape[0] // 2)
        lines_info = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if (abs(x1 - center[0]) < 30 and abs(y1 - center[1]) < 30) or (abs(x2 - center[0]) < 30 and abs(y2 - center[1]) < 30):
                    dist1 = np.hypot(x1 - center[0], y1 - center[1])
                    dist2 = np.hypot(x2 - center[0], y2 - center[1])
                    tip = (x1, y1) if dist1 > dist2 else (x2, y2)
                    dx = tip[0] - center[0]
                    dy = center[1] - tip[1]
                    angle = math.degrees(math.atan2(dy, dx))
                    length = max(dist1, dist2)
                    lines_info.append({'angle': angle, 'length': length, 'coords': (x1, y1, x2, y2)})

        # Фильтрация по углу
        def cluster_lines_by_angle(lines_info, threshold=25):
            clusters = []
            for line in lines_info:
                found = False
                for cluster in clusters:
                    if abs(line['angle'] - cluster['angle']) < threshold:
                        cluster['lines'].append(line)
                        found = True
                        break
                if not found:
                    clusters.append({'angle': line['angle'], 'lines': [line]})
            result = []
            for c in clusters:
                longest = max(c['lines'], key=lambda x: x['length'])
                result.append(longest)
            return result

        self.filtered_lines = cluster_lines_by_angle(lines_info)
        for line in self.filtered_lines:
            x1, y1, x2, y2 = line['coords']
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 3)

        self.show_intermediate(output, "Detected Clock Hands")

    def step_calculate_time(self):
        """6. Расчёт времени по углам стрелок и вывод на изображение."""
        if not self.filtered_lines:
            print("Стрелки не найдены.")
            return

        gray = cv2.cvtColor(self.result, cv2.COLOR_BGR2GRAY)
        center = (gray.shape[1] // 2, gray.shape[0] // 2)
        hand_angles = []

        for line in self.filtered_lines:
            x1, y1, x2, y2 = line['coords']
            dist1 = np.hypot(x1 - center[0], y1 - center[1])
            dist2 = np.hypot(x2 - center[0], y2 - center[1])
            tip = (x1, y1) if dist1 > dist2 else (x2, y2)
            dx = tip[0] - center[0]
            dy = center[1] - tip[1]
            angle = math.degrees(math.atan2(dy, dx))
            angle = (angle + 360) % 360
            angle_from_12 = (90 - angle) % 360
            hand_angles.append({'angle': angle_from_12, 'length': max(dist1, dist2)})

        # сортируем стрелки по длине
        hand_angles = sorted(hand_angles, key=lambda x: x['length'], reverse=True)

        if len(hand_angles) >= 2:
            minute = int(round(hand_angles[0]['angle'] / 6)) % 60
            hour = int(round(hand_angles[1]['angle'] / 30)) % 12

            # 👇 добавляем время на изображение
            output = self.result.copy()
            detected_time = f"{hour}:{minute:02d}"

            cv2.putText(
                output,
                detected_time,
                (20, 380),  # координаты текста
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,             # размер шрифта
                (0, 255, 0),     # зелёный цвет
                3,               # толщина линий
                cv2.LINE_AA
            )

            # показываем финальный результат
            self.result = output
            self.show_intermediate(output, f"Detected time: {detected_time}")

            # также выводим в окно
            messagebox.showinfo("Распознанное время", detected_time)
            print(f"Detected time: {detected_time}")

        else:
            print("Недостаточно стрелок для определения времени.")
