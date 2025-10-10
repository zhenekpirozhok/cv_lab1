from tkinter import simpledialog
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