import torch
import os
import cv2
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask

def create_line_mask_learn_from_create_mask(width, height, 
                               line_width=2,
                               angle=45, 
                               x=None, 
                               y=None):
    """
    生成一条线性区域的掩码，线性区域的值为1，其余区域为0。
    
    参数:
        width: 掩码宽度
        height: 掩码高度
        line_width: 线宽
        angle: 线的角度（单位：度）
        x: 线的起始横坐标（可选）
        y: 线的起始纵坐标（可选）
    
    返回:
        numpy数组: 线性缺失区域掩码，值为0（正常）或1（缺失）
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 如果未指定起始点，随机生成
    if x is None:
        x = random.randint(0, width)
    if y is None:
        y = random.randint(0, height)
    
    # 将角度转换为弧度
    angle_rad = np.radians(angle)
    cos_ang = np.cos(angle_rad)
    sin_ang = np.sin(angle_rad)
    
    # 创建网格坐标
    y_grid, x_grid = np.indices((height, width))
    
    # 计算点到线的距离
    distance = np.abs((x_grid - x) * sin_ang - (y_grid - y) * cos_ang)
    
    # 生成线性区域掩码
    line_mask = (distance < line_width / 2).astype(np.uint8)
    mask = np.maximum(mask, line_mask)
    print(mask.size())
    return mask


"""
def generate_stroke_mask(width, height, parts=10, maxVertex=20, maxLength=100, maxBrushWidth=600, maxAngle=360):
    # 1. 初始化掩码（形状为 (height, width)，避免冗余通道维度）
    mask = np.zeros((height, width), dtype=np.float32)  # 注意：width和height的顺序！
    
    for i in range(parts):
        # 2. 确保 np_free_form_mask 返回 (height, width) 形状的掩码
        stroke = np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, width, height)
        mask = mask + stroke
    
    # 3. 裁剪到 [0,1] 范围，转为 uint8 类型（0-255）
    mask = np.minimum(mask, 1.0)  # 确保不超过1.0
    mask = (mask * 255).astype(np.uint8)  # 转为 0-255 的 uint8
    
    # 4. 移除所有冗余维度（确保形状为 (height, width)）
    mask = np.squeeze(mask)
    
    return mask
"""
import numpy as np
import cv2
"""
def generate_stroke_mask(width, height, parts=10, maxVertex=20, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((width, height, 1), dtype=np.float32)
    print(mask.size)
    print("mask.size")
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, width, height)
    mask = np.minimum(mask, 1.0)  # Ensure values are in [0, 1]
    mask = np.transpose(mask, [2, 0, 1])  # Transpose to (1, height, width)
    mask = np.expand_dims(mask, 0)  # Add batch dimension: (1, 1, height, width)
    mask = np.repeat(mask, repeats=3, axis=1)  # Repeat for 3 channels: (1, 3, height, width)
    
    # Convert to uint8 for PIL compatibility
    mask = (mask * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    print(mask.size)
    return mask
"""



def generate_stroke_mask(width, height, num_strokes=5, stroke_width_range=(2, 10), randomize=True):
    """
    生成类似手写笔画的不规则掩码。

    参数:
        width: 掩码宽度
        height: 掩码高度
        num_strokes: 笔画数量
        stroke_width_range: 笔画宽度范围 (min, max)
        randomize: 是否随机化参数

    返回:
        numpy数组: 不规则缺失区域掩码，值为0（正常）或255（缺失）
    """
    
    # 初始化全0掩码（0表示正常区域）
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if randomize:
        # 随机化笔画数量
        num_strokes = random.randint(1, max(2, num_strokes))
        
        # 随机笔画宽度范围，确保第一个值小于第二个值
        min_width = random.randint(stroke_width_range[0], stroke_width_range[1] - 1)
        max_width = random.randint(min_width + 1, stroke_width_range[1])
        stroke_width_range = (min_width, max_width)
    
    # 确保 stroke_width_range 是有效的
    if stroke_width_range[0] > stroke_width_range[1]:
        raise ValueError(f"Invalid stroke_width_range: {stroke_width_range}. The first value must be less than or equal to the second value.")
    
    # 生成多条笔画
    for _ in range(num_strokes):
        # 随机生成控制点
        num_control_points = random.randint(3, 8)  # 控制点的数量
        control_points = []
        for _ in range(num_control_points):
            x = random.randint(0, width)
            y = random.randint(0, height)
            control_points.append((x, y))
        
        # 随机笔画宽度
        stroke_width = random.randint(stroke_width_range[0], stroke_width_range[1])
        
        # 使用贝塞尔曲线或样条曲线生成笔画
        stroke_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(stroke_mask, [np.array(control_points, dtype=np.int32)], isClosed=False, 
                      color=255, thickness=stroke_width)
        
        # 合并到掩码中
        mask = np.maximum(mask, stroke_mask)
    
    return mask
    
def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int64)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int64)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask

"""
def random_irregular_mask(img):
  
    transform = transforms.Compose([transforms.ToTensor()])
    #mask = torch.ones_like(img)
    size = img.size()
    mask = torch.ones(1, size[1], size[2])
    img = np.zeros((size[1], size[2], 1), np.uint8)

    # Set size scale
    max_width = 20
    if size[1] < 64 or size[2] < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    number = random.randint(16, 64)
    for _ in range(number):
        model = random.random()
        if model < 0.6:
            # Draw random lines
            x1, x2 = randint(1, size[1]), randint(1, size[1])
            y1, y2 = randint(1, size[2]), randint(1, size[2])
            thickness = randint(4, max_width)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        elif model > 0.6 and model < 0.8:
            # Draw random circles
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            radius = randint(4, max_width)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

        elif model > 0.8:
            # Draw random ellipses
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            s1, s2 = randint(1, size[1]), randint(1, size[2])
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(4, max_width)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    img = img.reshape(size[2], size[1])
    img = Image.fromarray(img*255)

    img_mask = transform(img)
    mask[0, :, :] = img_mask < 1

    return mask
"""

import numpy as np
import random
def create_irregular_mask(width, height,
                          num_curves=3,
                          curve_width_range=(4, 12),
                          amplitude_range=(20, 60),
                          wavelength_range=(50, 200),
                          randomize=True):
    # 初始化全0掩码（0表示正常区域）
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if randomize:
        # 随机化曲线数量
        num_curves = random.randint(1, max(2, num_curves))
        # 随机振幅和波长
        amplitude = random.uniform(*amplitude_range)
        wavelength = random.uniform(*wavelength_range)
    else:
        amplitude = (amplitude_range[0] + amplitude_range[1]) / 2
        wavelength = (wavelength_range[0] + wavelength_range[1]) / 2
    
    # 生成基本正弦曲线
    x = np.linspace(0, width, width)
    base_curve = amplitude * np.sin(2 * np.pi * x / wavelength)
    
    for i in range(num_curves):
        # 随机曲线宽度
        curve_width = random.randint(*curve_width_range) if randomize else (curve_width_range[0] + curve_width_range[1]) // 2
        
        # 随机相位偏移
        phase = random.uniform(0, 2 * np.pi) if randomize else 0
        
        # 生成当前曲线
        current_curve = amplitude * np.sin(2 * np.pi * x / wavelength + phase)
        
        # 计算曲线在图像中的位置
        y = np.linspace(0, height, height)
        Y, X = np.meshgrid(y, x)
        distance = np.abs(Y - (base_curve + current_curve))
        
        # 绘制曲线（距离小于曲线宽度的一半即为缺失区域）
        curve_mask = (distance < curve_width / 2).astype(np.uint8) * 255
        mask = np.maximum(mask, curve_mask)  # 合并曲线到掩码
    
    return mask
  
def create_line_mask(width, height, 
                     num_lines=3, 
                     line_width_range=(2, 8), 
                     angle_range=(0, 180), 
                     gap_range=(10, 50),
                     randomize=True):

        #创建包含多条线性缺失区域的掩码
    
        #参数:
            #width: 掩码宽度
            #height: 掩码高度
            #num_lines: 线条数量
            #line_width_range: 线条宽度范围 (min, max)
            #angle_range: 线条角度范围 (min, max) 单位：度
            #gap_range: 线条间距范围 (min, max) 单位：像素
            #randomize: 是否随机化参数
    
        #返回:
            #numpy数组: 线性缺失区域掩码，值为0（正常）或255（缺失）

        # 初始化全0掩码（0表示正常区域）
        mask = np.zeros((height, width), dtype=np.uint8)
    
        if randomize:
            # 随机化线条数量
            num_lines = random.randint(1, max(2, num_lines))
            # 随机主角度
            main_angle = random.uniform(*angle_range)
        else:
            main_angle = (angle_range[0] + angle_range[1]) / 2  # 使用平均角度
    
        # 计算角度的三角函数（用于线条定位）
        angle_rad = np.radians(main_angle)
        cos_ang = np.cos(angle_rad)
        sin_ang = np.sin(angle_rad)
    
        # 计算线条起始位置（图像中心附近）
        center_x, center_y = width // 2, height // 2
    
        for i in range(num_lines):
            # 随机线条宽度
            line_width = random.randint(*line_width_range) if randomize else (line_width_range[0] + line_width_range[1]) // 2
        
            # 计算线条偏移量（基于主角度排列线条）
            if num_lines > 1:
                gap = random.randint(*gap_range) if randomize else (gap_range[0] + gap_range[1]) // 2
                # 计算相对于中心的偏移（左右/上下分布）
                offset = (i - (num_lines - 1) / 2) * gap
            else:
                offset = 0
        
            # 计算线条在图像中的位置
            # 创建网格坐标
            y, x = np.indices((height, width))
            # 将坐标转换为相对中心的偏移
            x_rel = x - center_x
            y_rel = y - center_y
        
            # 计算点到主线的距离（用于绘制平行线）
            distance = np.abs(x_rel * sin_ang - y_rel * cos_ang - offset)
        
            # 绘制线条（距离小于线宽的一半即为缺失区域）
            line_mask = (distance < line_width / 2).astype(np.uint8) *255
            mask = np.maximum(mask, line_mask)  # 合并线条到掩码
            return mask
        


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)




class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)
