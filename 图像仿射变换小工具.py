import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from datetime import datetime
import os
import json

# 加载配置文件
def load_config():
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        messagebox.showerror("配置错误", f"加载配置文件失败: {str(e)}")
        return None

# 加载配置
config = load_config()
if not config:
    exit(1)

# 检查并创建data和results文件夹
for folder in [config['folders']['data'], config['folders']['results']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

class AdvancedImageBlender:
    def __init__(self, root):
        self.root = root
        self.root.title(config['window']['title'])
        
        # 获取屏幕尺寸
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 设置窗口大小为屏幕大小的比例
        window_width = int(screen_width * config['window']['size_ratio'])
        window_height = int(screen_height * config['window']['size_ratio'])
        
        # 计算窗口位置使其居中
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # 设置窗口大小和位置
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 状态变量初始化
        self.base_image = None
        self.base_image_original_size = (0, 0)  # 保存底层图像原始尺寸
        self.overlay_image = None
        self.working_image = None
        self.current_scale = 1.0
        self.current_angle = 0.0
        self.drag_start = None
        self.transform_history = []  # 历史记录
        self.redo_history = []  # 重做历史记录
        self.max_history = config['transform']['max_history']  # 最大历史记录数
        self.mouse_mode = "move"  # move/rotate/scale/shear
        
        # 融合参数
        self.blend_alpha = config['image']['default_alpha']  # 默认融合透明度
        
        # 缩放控制点状态
        self.resize_handles = {
            "right": None,
            "bottom": None,
            "corner": None
        }
        self.active_handle = None
        self.resize_start = None
        self.resize_start_scale = None
        
        # 拖动参数
        self.last_mouse_pos = (0, 0)
        
        # 界面初始化
        self.create_widgets()
        self.setup_bindings()
        
        # 视觉反馈参数
        self.cursor_feedback = None
        self.temp_line = None

        # 控制参数
        self.rotation_sensitivity = config['transform']['rotation_sensitivity']  # 旋转灵敏度系数
        self.shear_sensitivity = config['transform']['shear_sensitivity']  # 剪切灵敏度系数

        # 新增视觉元素
        self.rotation_indicator = None

    def create_widgets(self):
        """创建主界面组件"""
        # 工具栏
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # 操作按钮组
        control_frame = tk.Frame(toolbar)
        control_frame.pack(side=tk.LEFT, padx=5)
        
        self.btn_init_canvas = tk.Button(control_frame, text="画布初始化", command=self.initialize_canvas)
        self.btn_init_canvas.pack(side=tk.LEFT)
        
        self.btn_load_base = tk.Button(control_frame, text="加载底层", command=self.load_base_image)
        self.btn_load_base.pack(side=tk.LEFT, padx=5)
        
        self.btn_load_overlay = tk.Button(control_frame, text="加载上层", command=self.load_overlay_image)
        self.btn_load_overlay.pack(side=tk.LEFT, padx=5)
        
        self.btn_calculate = tk.Button(control_frame, text="计算矩阵", command=self.calculate_transform)
        self.btn_calculate.pack(side=tk.LEFT)
        
        self.btn_save = tk.Button(control_frame, text="保存图像", command=self.save_result)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        self.btn_verify_matrix = tk.Button(control_frame, text="验证矩阵", command=self.verify_transform_matrix)
        self.btn_verify_matrix.pack(side=tk.LEFT, padx=5)

        # 添加撤销和重做按钮
        self.btn_undo = tk.Button(control_frame, text="←", width=2, command=self.undo_transform)
        self.btn_undo.pack(side=tk.LEFT, padx=2)
        
        self.btn_redo = tk.Button(control_frame, text="→", width=2, command=self.redo_transform)
        self.btn_redo.pack(side=tk.LEFT, padx=2)
        
        # 模式选择组
        mode_frame = tk.Frame(toolbar)
        mode_frame.pack(side=tk.LEFT, padx=10)
        
        self.mode_var = tk.StringVar(value="move")
        tk.Radiobutton(mode_frame, text="移动", variable=self.mode_var, 
                      value="move", command=self.update_mode).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="旋转", variable=self.mode_var, 
                      value="rotate", command=self.update_mode).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="缩放", variable=self.mode_var, 
                      value="scale", command=self.update_mode).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="剪切", variable=self.mode_var, 
                      value="shear", command=self.update_mode).pack(side=tk.LEFT)
        
        # 添加剪切清零按钮
        self.btn_reset_shear = tk.Button(mode_frame, text="剪切清零", 
                                       command=self.reset_shear)
        self.btn_reset_shear.pack(side=tk.LEFT, padx=5)
        
        # 参数显示组
        info_frame = tk.Frame(toolbar)
        info_frame.pack(side=tk.RIGHT)
        
        self.lbl_position = tk.Label(info_frame, text="位置: (0, 0)")
        self.lbl_position.pack(side=tk.LEFT, padx=5)
        
        self.lbl_scale = tk.Label(info_frame, text="缩放: 100%")
        self.lbl_scale.pack(side=tk.LEFT, padx=5)
        
        self.lbl_angle = tk.Label(info_frame, text="角度: 0°")
        self.lbl_angle.pack(side=tk.LEFT, padx=5)

        # 透明度控制组
        alpha_frame = tk.Frame(toolbar)
        alpha_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(alpha_frame, text="透明度:").pack(side=tk.LEFT)
        self.alpha_scale = ttk.Scale(
            alpha_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            length=100,
            value=self.blend_alpha,
            command=self.update_blend
        )
        self.alpha_scale.pack(side=tk.LEFT, padx=5)
        
        # 主画布
        self.canvas = tk.Canvas(self.root, bg=config['window']['background_color'], cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 等待画布创建完成后再获取实际大小
        self.root.update_idletasks()

    def setup_bindings(self):
        """设置事件绑定"""
        self.canvas.bind("<ButtonPress-1>", self.start_interaction)
        self.canvas.bind("<B1-Motion>", self.process_interaction)
        self.canvas.bind("<ButtonRelease-1>", self.end_interaction)
        self.canvas.bind("<Motion>", self.update_cursor_feedback)
        self.root.bind("<Control-z>", self.undo_transform)
        
        # 添加鼠标进入/离开控制点的事件绑定
        self.canvas.bind("<Enter>", self.on_canvas_enter)
        self.canvas.bind("<Leave>", self.on_canvas_leave)

    def load_base_image(self):
        """加载底层图像"""
        try:
            path = filedialog.askopenfilename(
                initialdir="data",
                filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.gif")]
            )
            if not path:
                return

            # 加载原始图像并保存原始大小
            self.base_image = Image.open(path).convert("RGBA")
            self.base_image_original_size = self.base_image.size
            
            # 更新画布尺寸和显示
            self.canvas.config(width=self.base_image.width, height=self.base_image.height)
            self.canvas.delete("all")
            self.base_tk = ImageTk.PhotoImage(self.base_image)
            self.canvas.create_image(
                self.base_image.width // 2,
                self.base_image.height // 2,
                image=self.base_tk,
                anchor="center"
            )
            
            # 更新状态
            self.base_image_path = path
            self.update_status_labels()
            
        except Exception as e:
            messagebox.showerror("错误", f"加载底层图像失败: {str(e)}")

    def load_overlay_image(self):
        """加载上层图像"""
        try:
            path = filedialog.askopenfilename(
                initialdir="data",
                filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.gif")]
            )
            if not path:
                return

            # 加载并初始化上层图像
            original = Image.open(path).convert("RGBA")
            self.overlay_image = {
                "original": original,
                "position": (0, 0),
                "scale_x": 1.0,
                "scale_y": 1.0,
                "angle": 0.0,
                "shear_x": 0.0,
                "shear_y": 0.0
            }
            
            # 初始化工作图像和显示
            self.working_image = original.copy()
            self.overlay_tk = ImageTk.PhotoImage(self.working_image)
            self.canvas.config(width=original.width, height=original.height)
            self.redraw_canvas()
            
            # 更新状态
            self.overlay_image_path = path
            self.update_status_labels()
            
        except Exception as e:
            messagebox.showerror("错误", f"加载上层图像失败: {str(e)}")

    def update_mode(self):
        """更新操作模式"""
        self.mouse_mode = self.mode_var.get()
        # 清除所有视觉反馈
        self.canvas.delete("scale_feedback")
        self.canvas.delete("rotate_feedback")
        self.canvas.delete("text_feedback")
        self.canvas.delete("resize_handle")
        # 重绘画布以更新控制点显示
        self.redraw_canvas()

    def start_interaction(self, event):
        """开始交互操作"""
        self.drag_start = (event.x, event.y)
        self.last_mouse_pos = (event.x, event.y)
        self.canvas.delete("guide")

        if self.overlay_image:
            # 检查是否点击了缩放控制点
            handle = self.get_handle_at_position(event.x, event.y)
            if handle:
                self.active_handle = handle
                self.resize_start = (event.x, event.y)
                self.resize_start_scale = self.overlay_image["scale_x"]
                # 记录初始状态
                self.save_state()
                return
                
            # 记录初始状态
            self.save_state()
            
            if self.mouse_mode == "rotate":
                self.draw_rotation_guide(event)

    def process_interaction(self, event):
        """处理交互操作"""
        if not self.overlay_image or not self.drag_start:
            return
        
        current_pos = (event.x, event.y)
        
        # 根据当前模式处理交互
        if self.mouse_mode == "scale":
            if self.active_handle:
                self.handle_resize(event)
        elif self.mouse_mode == "move":
            # 直接使用鼠标移动的距离更新位置
            dx = current_pos[0] - self.last_mouse_pos[0]
            dy = current_pos[1] - self.last_mouse_pos[1]
            
            # 直接更新位置，不使用任何平滑或惯性
            new_x = self.overlay_image["position"][0] + dx
            new_y = self.overlay_image["position"][1] + dy
            self.overlay_image["position"] = (new_x, new_y)
        elif self.mouse_mode == "rotate":
            self.handle_rotate(event, current_pos[0] - self.last_mouse_pos[0])
        elif self.mouse_mode == "shear":
            self.handle_shear(event)
        
        self.last_mouse_pos = current_pos
        self.update_overlay()
        self.redraw_canvas()
        self.update_status_labels()
        self.update_visual_feedback()

    def handle_resize(self, event):
        """处理缩放控制点的拖动"""
        if not self.resize_start or not self.resize_start_scale:
            return
            
        dx = event.x - self.resize_start[0]
        dy = event.y - self.resize_start[1]
        
        # 根据不同的控制点计算缩放比例
        if self.active_handle == "right":
            # 只进行水平缩放
            scale_factor = 1 + (dx / (self.working_image.width * self.resize_start_scale)) * 1.5
            self.overlay_image["scale_x"] = self.resize_start_scale * max(
                config['transform']['scale_limits']['min'],
                min(config['transform']['scale_limits']['max'], scale_factor)
            )
            self.overlay_image["scale_y"] = self.resize_start_scale
        elif self.active_handle == "bottom":
            # 只进行垂直缩放
            scale_factor = 1 + (dy / (self.working_image.height * self.resize_start_scale)) * 1.5
            self.overlay_image["scale_x"] = self.resize_start_scale
            self.overlay_image["scale_y"] = self.resize_start_scale * max(
                config['transform']['scale_limits']['min'],
                min(config['transform']['scale_limits']['max'], scale_factor)
            )
        else:  # corner
            # 同时进行水平和垂直缩放
            scale_x = 1 + (dx / (self.working_image.width * self.resize_start_scale)) * 1.5
            scale_y = 1 + (dy / (self.working_image.height * self.resize_start_scale)) * 1.5
            scale_factor = max(scale_x, scale_y)
            scale_factor = max(
                config['transform']['scale_limits']['min'],
                min(config['transform']['scale_limits']['max'], scale_factor)
            )
            self.overlay_image["scale_x"] = self.resize_start_scale * scale_factor
            self.overlay_image["scale_y"] = self.resize_start_scale * scale_factor

    def handle_rotate(self, event, delta_x):
        """处理旋转操作"""
        if not self.overlay_image:
            return
        
        # 获取图像实际中心（考虑缩放后的位置）
        center = (
            self.overlay_image["position"][0] + 
            self.working_image.width * 0.5,
            self.overlay_image["position"][1] + 
            self.working_image.height * 0.5
        )
        
        # 计算旋转向量
        vec_start = np.array([self.drag_start[0]-center[0], 
                            self.drag_start[1]-center[1]])
        vec_current = np.array([event.x-center[0], 
                            event.y-center[1]])
        
        # 使用叉积计算精确角度差
        angle_rad = np.arctan2(
            np.cross(vec_start, vec_current),
            np.dot(vec_start, vec_current)
        )
        angle_delta = np.degrees(angle_rad)
        
        # 降低旋转灵敏度
        self.overlay_image["angle"] += angle_delta * 0.2  # 降低到0.3倍
        self.overlay_image["angle"] %= 360  # 角度标准化

    def calculate_angle_delta(self, v1, v2):
        """计算带方向的精确角度差"""
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        det = v1[0]*v2[1] - v1[1]*v2[0]
        return np.degrees(np.arctan2(det, dot))

    def update_visual_feedback(self):
        """多级视觉反馈"""
        self.canvas.delete("scale_feedback")
        self.canvas.delete("rotate_feedback")
        self.canvas.delete("text_feedback")

        if not self.overlay_image:
            return

        # 获取图像实际显示尺寸
        img_w, img_h = self.working_image.size if self.working_image else (0, 0)
        center_x = self.overlay_image["position"][0] + img_w // 2
        center_y = self.overlay_image["position"][1] + img_h // 2

        # 旋转指示器（红色十字）
        if self.mouse_mode == "rotate":
            # 红色十字线
            self.canvas.create_line(
                center_x - config['ui']['rotation_guide']['length'], center_y,
                center_x + config['ui']['rotation_guide']['length'], center_y,
                fill=config['ui']['guide_color'],
                width=config['ui']['rotation_guide']['width'],
                tags="rotate_feedback"
            )
            self.canvas.create_line(
                center_x, center_y - config['ui']['rotation_guide']['length'],
                center_x, center_y + config['ui']['rotation_guide']['length'],
                fill=config['ui']['guide_color'],
                width=config['ui']['rotation_guide']['width'],
                tags="rotate_feedback"
            )

        # 实时参数显示（白色文本）
        info_text = f"缩放: {self.overlay_image['scale_x']:.2f}x\n角度: {self.overlay_image['angle']:.1f}°"
        self.canvas.create_text(
            self.last_mouse_pos[0] + config['ui']['feedback']['offset_x'],
            self.last_mouse_pos[1] + config['ui']['feedback']['offset_y'],
            text=info_text,
            fill=config['ui']['text_color'],
            anchor=tk.NW,
            font=tuple(config['ui']['text_font']),
            tags="text_feedback"
        )

    def end_interaction(self, event):
        """结束交互"""
        self.drag_start = None
        self.active_handle = None
        self.resize_start = None
        self.resize_start_scale = None
        self.canvas.delete("guide")

    def update_overlay(self):
        """更新上层图像变换"""
        if not self.overlay_image:
            return

        try:
            img = self.overlay_image["original"]
            
            # 1. 缩放处理
            w = int(img.width * self.overlay_image["scale_x"])
            h = int(img.height * self.overlay_image["scale_y"])
            scaled = img.resize((w, h), Image.Resampling.LANCZOS)
            
            # 2. 剪切处理
            if self.overlay_image["shear_x"] != 0 or self.overlay_image["shear_y"] != 0:
                shear_matrix = [
                    1, self.overlay_image["shear_x"], 0,
                    self.overlay_image["shear_y"], 1, 0
                ]
                scaled = scaled.transform(
                    scaled.size,
                    Image.AFFINE,
                    shear_matrix,
                    Image.BICUBIC
                )
            
            # 3. 旋转处理
            rotated = scaled.rotate(
                -self.overlay_image["angle"],
                expand=True,
                resample=Image.BICUBIC,
                center=(w/2, h/2)
            )

            # 4. 更新工作图像
            self.working_image = rotated.convert("RGBA")
            self.overlay_tk = ImageTk.PhotoImage(self.working_image)

        except Exception as e:
            messagebox.showerror("图像处理错误", f"更新上层图像失败: {str(e)}")

    def redraw_canvas(self):
        """重绘画布"""
        self.canvas.delete("all")
        
        # 绘制底层图像
        if self.base_image:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.base_tk)
        
        # 绘制上层图像
        if self.working_image:
            # 调整上层图像的透明度
            overlay = self.working_image.copy()
            if self.blend_alpha < 1.0:
                overlay.putalpha(int(255 * self.blend_alpha))
            
            # 创建新的PhotoImage对象以保持引用
            overlay_img = ImageTk.PhotoImage(overlay)
            self.canvas.overlay_image = overlay_img  # 保持引用
            
            x, y = self.overlay_image["position"]
            self.canvas.create_image(x, y, anchor=tk.NW, image=overlay_img)
            
            # 在缩放模式下绘制控制点
            if self.mouse_mode == "scale":
                self.draw_resize_handles()

    def draw_resize_handles(self):
        """绘制缩放控制点"""
        if not self.overlay_image or not self.working_image:
            return
            
        x, y = self.overlay_image["position"]
        w, h = self.working_image.size
        
        # 绘制右侧控制点
        self.canvas.create_rectangle(
            x + w - config['ui']['handle_size'], y + h/2 - config['ui']['handle_size'],
            x + w + config['ui']['handle_size'], y + h/2 + config['ui']['handle_size'],
            fill=config['ui']['handle_color']['fill'],
            outline=config['ui']['handle_color']['outline'],
            tags="resize_handle"
        )
        
        # 绘制底部控制点
        self.canvas.create_rectangle(
            x + w/2 - config['ui']['handle_size'], y + h - config['ui']['handle_size'],
            x + w/2 + config['ui']['handle_size'], y + h + config['ui']['handle_size'],
            fill=config['ui']['handle_color']['fill'],
            outline=config['ui']['handle_color']['outline'],
            tags="resize_handle"
        )
        
        # 绘制右下角控制点
        self.canvas.create_rectangle(
            x + w - config['ui']['handle_size'], y + h - config['ui']['handle_size'],
            x + w + config['ui']['handle_size'], y + h + config['ui']['handle_size'],
            fill=config['ui']['handle_color']['fill'],
            outline=config['ui']['handle_color']['outline'],
            tags="resize_handle"
        )

    def on_canvas_enter(self, event):
        """鼠标进入画布时的处理"""
        self.update_cursor_feedback(event)

    def on_canvas_leave(self, event):
        """鼠标离开画布时的处理"""
        self.canvas.config(cursor="crosshair")

    def update_status_labels(self):
        """更新状态标签"""
        if not self.overlay_image:
            return
            
        x, y = self.overlay_image["position"]
        self.lbl_position.config(text=f"位置: ({x}, {y})")
        self.lbl_scale.config(text=f"缩放: X{self.overlay_image['scale_x']*100:.1f}% Y{self.overlay_image['scale_y']*100:.1f}%")
        self.lbl_angle.config(text=f"角度: {self.overlay_image['angle']:.1f}°")

    def calculate_transform(self):
        """计算仿射变换矩阵"""
        try:
            if not self.base_image or not self.overlay_image:
                raise ValueError("请先加载底层和上层图像")
                
            # 获取变换参数
            try:
                sx = self.overlay_image["scale_x"]
                sy = self.overlay_image["scale_y"]
                theta = np.radians(self.overlay_image["angle"])
                tx, ty = self.overlay_image["position"]
                shear_x = self.overlay_image["shear_x"]
                shear_y = self.overlay_image["shear_y"]
            except KeyError as e:
                raise ValueError(f"缺少必要的变换参数: {str(e)}")
            except Exception as e:
                raise ValueError(f"获取变换参数失败: {str(e)}")
            
            # 构建变换矩阵
            try:
                rotate_matrix = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                
                scale_matrix = np.array([
                    [sx, 0, 0],
                    [0, sy, 0],
                    [0, 0, 1]
                ])
                
                shear_matrix = np.array([
                    [1, shear_x, 0],
                    [shear_y, 1, 0],
                    [0, 0, 1]
                ])
                
                translate_matrix = np.array([
                    [1, 0, tx],
                    [0, 1, ty],
                    [0, 0, 1]
                ])
                
                # 组合变换矩阵：先缩放 -> 剪切 -> 旋转 -> 平移
                affine_matrix = translate_matrix @ rotate_matrix @ shear_matrix @ scale_matrix
            except Exception as e:
                raise ValueError(f"构建变换矩阵失败: {str(e)}")
            
            # 显示结果
            self.show_matrix_result(affine_matrix)
            
        except Exception as e:
            messagebox.showerror("计算错误", str(e))

    def show_matrix_result(self, matrix):
        """显示矩阵结果窗口"""
        result_win = tk.Toplevel(self.root)
        result_win.title("仿射变换矩阵")
        
        # 设置窗口大小
        window_width = 400
        window_height = 200
        
        # 获取主窗口位置和大小
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        main_width = self.root.winfo_width()
        main_height = self.root.winfo_height()
        
        # 计算弹出窗口的位置，使其居中
        x = main_x + (main_width - window_width) // 2
        y = main_y + (main_height - window_height) // 2
        
        # 设置窗口位置和大小
        result_win.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        text = tk.Text(result_win, width=40, height=10)
        text.pack(padx=10, pady=10)
        
        # 格式化输出
        matrix_str = "完整3x3矩阵：\n"
        matrix_str += "\n".join(["[{:.4f}, {:.4f}, {:.4f}]".format(*row) for row in matrix])
        
        opencv_str = "\n\nOpenCV 2x3格式：\n"
        opencv_str += "[{:.4f}, {:.4f}, {:.4f}\n {:.4f}, {:.4f}, {:.4f}]".format(
            matrix[0,0], matrix[0,1], matrix[0,2],
            matrix[1,0], matrix[1,1], matrix[1,2]
        )
        
        text.insert(tk.END, matrix_str + opencv_str)
        text.config(state=tk.DISABLED)
        
        # 按钮框架
        btn_frame = tk.Frame(result_win)
        btn_frame.pack(pady=5)
        
        # 复制按钮
        copy_btn = tk.Button(btn_frame, text="复制到剪贴板",
                           command=lambda: self.copy_matrix(matrix))
        copy_btn.pack(side=tk.LEFT, padx=5)
        
        # 保存按钮
        save_btn = tk.Button(btn_frame, text="保存矩阵",
                           command=lambda: self.save_matrix_to_file(matrix))
        save_btn.pack(side=tk.LEFT, padx=30)  # 增加按钮之间的间距

    def copy_matrix(self, matrix):
        """复制矩阵到剪贴板"""
        self.root.clipboard_clear()
        self.root.clipboard_append(
            f"[[{matrix[0,0]:.4f}, {matrix[0,1]:.4f}, {matrix[0,2]:.4f}],\n"
            f" [{matrix[1,0]:.4f}, {matrix[1,1]:.4f}, {matrix[1,2]:.4f}]]"
        )
        messagebox.showinfo("已复制", "矩阵已复制到剪贴板")

    def save_matrix_to_file(self, matrix):
        """保存矩阵到文件"""
        try:
            # 计算逆矩阵
            M_inv = np.linalg.inv(matrix)
            
            # 格式化矩阵数据
            decimal_places = config['file']['matrix_format']['decimal_places']
            M_formatted = [[f"{x:.{decimal_places}f}" for x in row] for row in matrix.tolist()]
            M_inv_formatted = [[f"{x:.{decimal_places}f}" for x in row] for row in M_inv.tolist()]
            
            # 构建JSON数据
            base_size = self.base_image.size if self.base_image else (0, 0)
            overlay_size = self.overlay_image["original"].size if self.overlay_image else (0, 0)
            matrix_data = {
                "M": [float(x) for row in M_formatted for x in row],
                "M_inv": [float(x) for row in M_inv_formatted for x in row],
                "M_str": f"[{M_formatted[0][0]}, {M_formatted[0][1]}, {M_formatted[0][2]}]\n[{M_formatted[1][0]}, {M_formatted[1][1]}, {M_formatted[1][2]}]\n[{M_formatted[2][0]}, {M_formatted[2][1]}, {M_formatted[2][2]}]",
                "image_sizes": {
                    "base": {"width": base_size[0], "height": base_size[1]},
                    "overlay": {"width": overlay_size[0], "height": overlay_size[1]}
                }
            }
            
            # 生成文件名和保存路径
            timestamp = datetime.now().strftime(config['file']['matrix_format']['timestamp_format'])
            results_dir = config['folders']['results']
            os.makedirs(results_dir, exist_ok=True)
            default_filename = f"{timestamp}_affine_matrix.json"
            
            # 保存文件
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json")],
                initialfile=default_filename,
                initialdir=results_dir
            )
            
            if path:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(matrix_data, f, indent=4)
                messagebox.showinfo("保存成功", f"矩阵已保存至：{path}")
                
        except np.linalg.LinAlgError:
            messagebox.showerror("保存失败", "无法计算矩阵的逆矩阵，请检查矩阵是否可逆")
        except KeyError as e:
            messagebox.showerror("保存失败", f"配置文件缺少必要的键值: {str(e)}")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存过程中出错: {str(e)}")

    def save_result(self):
        """保存合成结果"""
        if not self.base_image or not self.overlay_image:
            messagebox.showerror("错误", "请先加载图像")
            return
            
        try:
            # 确保results目录存在
            results_dir = config['folders']['results']
            os.makedirs(results_dir, exist_ok=True)
            
            # 创建图像副本
            try:
                base = self.base_image.copy()
                overlay = self.working_image.copy()
            except Exception as e:
                messagebox.showerror("错误", f"创建图像副本失败: {str(e)}")
                return
            
            # 调整上层图像的透明度
            try:
                if self.blend_alpha < 1.0:
                    overlay.putalpha(int(255 * self.blend_alpha))
            except Exception as e:
                messagebox.showerror("错误", f"调整透明度失败: {str(e)}")
                return
            
            # 创建最终图像
            try:
                x, y = self.overlay_image["position"]
                final_image = base.copy()
                final_image.alpha_composite(overlay, (int(x), int(y)))
            except Exception as e:
                messagebox.showerror("错误", f"图像合成失败: {str(e)}")
                return
            
            # 生成默认文件名
            timestamp = datetime.now().strftime(config['file']['matrix_format']['timestamp_format'])
            default_filename = f"{timestamp}_blended.{config['file']['default_save_format']}"
            
            # 保存融合图像
            try:
                path = filedialog.asksaveasfilename(
                    defaultextension=f".{config['file']['default_save_format']}",
                    filetypes=[("PNG文件", "*.png"), ("JPEG文件", "*.jpg")],
                    initialfile=default_filename,
                    initialdir=results_dir
                )
                
                if path:
                    final_image.convert("RGB").save(path)
                    messagebox.showinfo("保存成功", f"融合图像已保存至：{path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存文件失败: {str(e)}")
                
        except Exception as e:
            messagebox.showerror("错误", f"保存过程出错: {str(e)}")

    def undo_transform(self, event=None):
        """撤销上一次变换"""
        if self.transform_history:
            # 保存当前状态到重做历史
            if self.overlay_image:
                current_state = {
                    "position": self.overlay_image["position"],
                    "scale_x": self.overlay_image["scale_x"],
                    "scale_y": self.overlay_image["scale_y"],
                    "angle": self.overlay_image["angle"],
                    "shear_x": self.overlay_image["shear_x"],
                    "shear_y": self.overlay_image["shear_y"]
                }
                self.redo_history.append(current_state)
            
            # 恢复到上一个状态
            prev_state = self.transform_history.pop()
            self.overlay_image.update(prev_state)
            self.update_overlay()
            self.redraw_canvas()
            self.update_status_labels()

    def redo_transform(self, event=None):
        """重做上一次撤销的变换"""
        if self.redo_history:
            # 保存当前状态到历史记录
            if self.overlay_image:
                current_state = {
                    "position": self.overlay_image["position"],
                    "scale_x": self.overlay_image["scale_x"],
                    "scale_y": self.overlay_image["scale_y"],
                    "angle": self.overlay_image["angle"],
                    "shear_x": self.overlay_image["shear_x"],
                    "shear_y": self.overlay_image["shear_y"]
                }
                self.transform_history.append(current_state)
                
                # 限制历史记录数量
                if len(self.transform_history) > self.max_history:
                    self.transform_history.pop(0)
            
            # 恢复到重做历史中的状态
            next_state = self.redo_history.pop()
            self.overlay_image.update(next_state)
            self.update_overlay()
            self.redraw_canvas()
            self.update_status_labels()

    def get_image_center(self):
        """获取上层图像中心坐标"""
        if not self.overlay_image or not self.working_image:
            return (0, 0)
        x, y = self.overlay_image["position"]
        w, h = self.working_image.size          # 使用变换后的尺寸
        return (x + w/2, y + h/2)

    def draw_rotation_guide(self, event):
        """绘制旋转辅助线"""
        center = self.get_image_center()
        self.canvas.delete("guide")
        self.canvas.create_line(
            center[0]-20, center[1], center[0]+20, center[1],
            fill="#FF0000", tags="guide"
        )
        self.canvas.create_line(
            center[0], center[1]-20, center[0], center[1]+20,
            fill="#FF0000", tags="guide"
        )

    def update_cursor_feedback(self, event):
        """更新光标反馈"""
        if not self.overlay_image:
            self.canvas.config(cursor="crosshair")
            return
            
        if self.mouse_mode == "scale":
            handle = self.get_handle_at_position(event.x, event.y)
            if handle:
                if handle == "right":
                    self.canvas.config(cursor="sb_h_double_arrow")
                elif handle == "bottom":
                    self.canvas.config(cursor="sb_v_double_arrow")
                else:  # corner
                    self.canvas.config(cursor="size_nw_se")
            else:
                self.canvas.config(cursor="crosshair")
        elif self.mouse_mode == "move":
            if self.is_over_image(event.x, event.y):
                self.canvas.config(cursor="hand2")
            else:
                self.canvas.config(cursor="crosshair")
        elif self.mouse_mode == "rotate":
            self.canvas.config(cursor="crosshair")
        elif self.mouse_mode == "shear":
            self.canvas.config(cursor="crosshair")

    def is_over_image(self, x, y):
        """判断坐标是否在上层图像范围内"""
        if not self.overlay_image:
            return False
        img_x, img_y = self.overlay_image["position"]
        w, h = self.working_image.size
        return (img_x <= x <= img_x + w and 
                img_y <= y <= img_y + h)

    def get_handle_at_position(self, x, y):
        """获取指定位置的控制点"""
        if not self.overlay_image or not self.working_image:
            return None
            
        img_x, img_y = self.overlay_image["position"]
        img_w, img_h = self.working_image.size
        
        # 定义控制点的区域（8像素的点击区域）
        handle_size = 8
        
        # 检查右侧控制点
        if (img_x + img_w - handle_size <= x <= img_x + img_w + handle_size and
            img_y + img_h/2 - handle_size <= y <= img_y + img_h/2 + handle_size):
            return "right"
            
        # 检查底部控制点
        if (img_x + img_w/2 - handle_size <= x <= img_x + img_w/2 + handle_size and
            img_y + img_h - handle_size <= y <= img_y + img_h + handle_size):
            return "bottom"
            
        # 检查右下角控制点
        if (img_x + img_w - handle_size <= x <= img_x + img_w + handle_size and
            img_y + img_h - handle_size <= y <= img_y + img_h + handle_size):
            return "corner"
            
        return None

    def update_blend(self, value):
        """更新透明度参数"""
        self.blend_alpha = float(value)
        self.redraw_canvas()

    def initialize_canvas(self):
        """初始化画布，清除所有图像"""
        # 清除所有图像数据
        self.base_image = None
        self.overlay_image = None
        self.working_image = None
        self.current_scale = 1.0
        self.current_angle = 0.0
        self.drag_start = None
        self.transform_history = []
        self.redo_history = []
        
        # 重置融合参数
        self.blend_alpha = 0.5
        self.alpha_scale.set(self.blend_alpha)
        
        # 重置缩放控制点状态
        self.resize_handles = {
            "right": None,
            "bottom": None,
            "corner": None
        }
        self.active_handle = None
        self.resize_start = None
        self.resize_start_scale = None
        
        # 重置拖动参数
        self.last_mouse_pos = (0, 0)
        
        # 清除画布上的所有内容
        self.canvas.delete("all")
        
        # 重置右上角状态标签
        self.lbl_position.config(text="位置: (0, 0)")
        self.lbl_scale.config(text="缩放: 100%")
        self.lbl_angle.config(text="角度: 0°")

    def verify_transform_matrix(self):
        """验证仿射变换矩阵"""
        if not self.base_image or not self.overlay_image:
            messagebox.showerror("错误", "请先加载底层和上层图像")
            return

        # 创建ROI选择窗口，使用底层图像
        roi_selector = ROISelector(self.root, self.base_image, self.base_image_original_size)
        self.roi = roi_selector.roi  # 保存ROI信息到类属性
        
        # 显示ROI信息窗口
        roi_win = tk.Toplevel(self.root)
        roi_win.title("ROI区域信息")
        
        # 设置窗口大小
        window_width = 400
        window_height = 200
        
        # 获取主窗口位置和大小
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        main_width = self.root.winfo_width()
        main_height = self.root.winfo_height()
        
        # 计算弹出窗口的位置，使其居中
        x = main_x + (main_width - window_width) // 2
        y = main_y + (main_height - window_height) // 2
        
        # 设置窗口位置和大小
        roi_win.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        text = tk.Text(roi_win, width=40, height=10)
        text.pack(padx=10, pady=10)
        
        # 格式化输出
        roi_str = "ROI区域坐标：\n"
        roi_str += f"[{self.roi[0]}, {self.roi[1]}, {self.roi[2]}, {self.roi[3]}]\n\n"
        roi_str += f"区域大小：\n"
        roi_str += f"宽度: {self.roi[2] - self.roi[0]} 像素\n"
        roi_str += f"高度: {self.roi[3] - self.roi[1]} 像素"
        
        text.insert(tk.END, roi_str)
        text.config(state=tk.DISABLED)
        
        # 按钮框架
        btn_frame = tk.Frame(roi_win)
        btn_frame.pack(pady=5)
        
        def show_matrix_input():
            roi_win.destroy()
            self.show_matrix_input_window()
        
        # 确认按钮
        confirm_btn = tk.Button(btn_frame, text="确认",
                              command=show_matrix_input)
        confirm_btn.pack(side=tk.LEFT, padx=30)
        
        # 等待ROI信息窗口关闭
        self.root.wait_window(roi_win)

    def show_matrix_input_window(self):
        """显示矩阵输入窗口"""
        # 创建输入窗口
        input_win = tk.Toplevel(self.root)
        input_win.title("输入仿射变换矩阵")
        input_win.geometry("400x300")
        
        # 获取主窗口位置和大小
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        main_width = self.root.winfo_width()
        main_height = self.root.winfo_height()
        
        # 计算弹出窗口的位置，使其居中
        x = main_x + (main_width - 400) // 2
        y = main_y + (main_height - 300) // 2
        
        # 设置窗口位置
        input_win.geometry(f"400x300+{x}+{y}")
        
        # 创建文本输入框
        text_frame = tk.Frame(input_win)
        text_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        tk.Label(text_frame, text="请输入3x3仿射变换矩阵（每行一个数组，用方括号包围）：").pack(anchor=tk.W)
        
        text_input = tk.Text(text_frame, height=10, width=40)
        text_input.pack(pady=5)
        text_input.insert(tk.END, "[1.0000, 0.0000, 0.0000]\n[0.0000, 1.0000, 0.0000]\n[0.0000, 0.0000, 1.0000]")  # 默认单位矩阵
        
        def apply_matrix():
            try:
                # 获取输入的矩阵
                matrix_text = text_input.get("1.0", tk.END).strip()
                matrix = []
                
                # 处理每一行
                for line in matrix_text.split('\n'):
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    
                    # 移除方括号并分割数值
                    line = line.strip('[]')
                    values = []
                    for val in line.split(','):
                        val = val.strip()
                        if val:  # 只处理非空值
                            try:
                                values.append(float(val))
                            except ValueError:
                                raise ValueError(f"无效的数值: {val}")
                    
                    if len(values) != 3:
                        raise ValueError(f"每行必须包含3个数值，当前行: {line}")
                    matrix.append(values)
                
                if len(matrix) != 3:
                    raise ValueError("矩阵必须包含3行")
                
                # 转换为numpy数组
                M = np.array(matrix)
                
                # 从矩阵中提取变换参数
                # 平移参数
                tx, ty = M[0:2, 2]
                
                # 缩放参数
                sx = np.sqrt(M[0,0]**2 + M[0,1]**2)
                sy = np.sqrt(M[1,0]**2 + M[1,1]**2)
                
                # 旋转角度
                theta = np.degrees(np.arctan2(M[1,0], M[0,0]))
                
                # 更新上层图像参数
                self.overlay_image["position"] = (tx, ty)
                self.overlay_image["scale_x"] = sx
                self.overlay_image["scale_y"] = sy
                self.overlay_image["angle"] = theta
                
                # 更新显示
                self.update_overlay()
                self.redraw_canvas()
                self.update_status_labels()
                
                # 关闭输入窗口
                input_win.destroy()
                
                messagebox.showinfo("成功", "矩阵已应用")
                
            except ValueError as e:
                messagebox.showerror("错误", f"矩阵格式错误：{str(e)}")
            except Exception as e:
                messagebox.showerror("错误", f"处理矩阵时出错：{str(e)}")
        
        # 添加确认按钮
        btn_frame = tk.Frame(input_win)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="确认", command=apply_matrix).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="取消", command=input_win.destroy).pack(side=tk.LEFT, padx=5)

    def save_state(self):
        """保存当前状态到历史记录"""
        if self.overlay_image:
            # 创建当前状态的副本
            current_state = {
                "position": self.overlay_image["position"],
                "scale_x": self.overlay_image["scale_x"],
                "scale_y": self.overlay_image["scale_y"],
                "angle": self.overlay_image["angle"],
                "shear_x": self.overlay_image["shear_x"],
                "shear_y": self.overlay_image["shear_y"]
            }
            
            # 添加到历史记录
            self.transform_history.append(current_state)
            
            # 限制历史记录数量
            if len(self.transform_history) > self.max_history:
                self.transform_history.pop(0)
            
            # 清空重做历史
            self.redo_history.clear()

    def handle_shear(self, event):
        """处理剪切变换"""
        if not self.overlay_image:
            return
            
        dx = event.x - self.last_mouse_pos[0]
        dy = event.y - self.last_mouse_pos[1]
        
        # 根据鼠标移动方向更新剪切参数
        self.overlay_image["shear_x"] += dx * self.shear_sensitivity
        self.overlay_image["shear_y"] += dy * self.shear_sensitivity
        
        # 限制剪切范围
        self.overlay_image["shear_x"] = max(
            config['transform']['shear_limits']['min'],
            min(config['transform']['shear_limits']['max'], self.overlay_image["shear_x"])
        )
        self.overlay_image["shear_y"] = max(
            config['transform']['shear_limits']['min'],
            min(config['transform']['shear_limits']['max'], self.overlay_image["shear_y"])
        )

    def reset_shear(self):
        """重置剪切参数"""
        if self.overlay_image:
            # 保存当前状态
            self.save_state()
            
            # 重置剪切参数
            self.overlay_image["shear_x"] = 0.0
            self.overlay_image["shear_y"] = 0.0
            
            # 更新显示
            self.update_overlay()
            self.redraw_canvas()
            self.update_status_labels()

class ROISelector:
    def __init__(self, parent, image, original_size):
        self.window = tk.Toplevel(parent)
        self.window.title("选择ROI区域 [仅用于区域提取, 不用于实际的验证矩阵]")
        
        # 保存原始图像和原始尺寸
        self.image = image
        self.original_size = original_size
        
        # 获取主窗口位置和大小
        main_x = parent.winfo_x()
        main_y = parent.winfo_y()
        main_width = parent.winfo_width()
        main_height = parent.winfo_height()
        
        # 计算窗口大小（考虑图像大小和边距）
        padding = 40  # 窗口边距
        button_height = 60  # 按钮区域高度
        
        # 计算缩放比例以适应窗口，保持原始宽高比
        width_ratio = (main_width * 0.9 - padding * 2) / image.width
        height_ratio = (main_height * 0.9 - padding * 2 - button_height) / image.height
        self.scale_ratio = min(width_ratio, height_ratio)
        
        # 计算缩放后的图像尺寸
        self.scaled_width = int(image.width * self.scale_ratio)
        self.scaled_height = int(image.height * self.scale_ratio)
        
        # 计算窗口大小
        window_width = self.scaled_width + padding * 2
        window_height = self.scaled_height + padding * 2 + button_height
        
        # 计算窗口位置使其居中
        x = main_x + (main_width - window_width) // 2
        y = main_y + (main_height - window_height) // 2
        
        # 设置窗口大小和位置
        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 创建画布框架
        canvas_frame = tk.Frame(self.window)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=padding//2, pady=padding//2)
        
        # 创建画布
        self.canvas = tk.Canvas(canvas_frame, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 缩放图像
        self.scaled_image = image.resize((self.scaled_width, self.scaled_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.scaled_image)
        
        # 在画布上显示图像，并居中显示
        self.x_offset = (window_width - self.scaled_width) // 2
        self.y_offset = (window_height - self.scaled_height - button_height) // 2
        
        # 等待画布创建完成
        self.window.update_idletasks()
        
        # 在画布上显示图像
        self.canvas.create_image(
            self.x_offset, self.y_offset,
            anchor=tk.NW,
            image=self.photo
        )
        
        # 设置画布滚动区域
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        # ROI选择变量
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.roi = None
        
        # 绑定鼠标事件
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # 添加确定按钮
        self.btn_frame = tk.Frame(self.window)
        self.btn_frame.pack(side=tk.BOTTOM, pady=10)
        
        self.btn_confirm = tk.Button(self.btn_frame, text="确定", command=self.confirm)
        self.btn_confirm.pack(side=tk.LEFT, padx=20)
        
        # 添加提示标签
        self.label = tk.Label(self.window, text="点击并拖动鼠标选择ROI区域，或直接点击确定使用全图")
        self.label.pack(side=tk.BOTTOM, pady=5)
        
        # 等待窗口关闭
        self.window.transient(parent)
        self.window.grab_set()
        parent.wait_window(self.window)
    
    def on_press(self, event):
        # 清除之前的矩形
        if self.rect:
            self.canvas.delete(self.rect)
        
        # 记录起始点（考虑偏移）
        self.start_x = self.canvas.canvasx(event.x) - self.x_offset
        self.start_y = self.canvas.canvasy(event.y) - self.y_offset
        
        # 创建新矩形
        self.rect = self.canvas.create_rectangle(
            self.start_x + self.x_offset,
            self.start_y + self.y_offset,
            self.start_x + self.x_offset,
            self.start_y + self.y_offset,
            outline='red', width=2
        )
    
    def on_drag(self, event):
        # 更新矩形大小（考虑偏移）
        cur_x = self.canvas.canvasx(event.x) - self.x_offset
        cur_y = self.canvas.canvasy(event.y) - self.y_offset
        self.canvas.coords(
            self.rect,
            self.start_x + self.x_offset,
            self.start_y + self.y_offset,
            cur_x + self.x_offset,
            cur_y + self.y_offset
        )
    
    def on_release(self, event):
        # 计算ROI区域（考虑偏移）
        end_x = self.canvas.canvasx(event.x) - self.x_offset
        end_y = self.canvas.canvasy(event.y) - self.y_offset
        
        # 确保坐标是有序的
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)
        
        # 将画布坐标转换回原始图像坐标
        # 使用原始图像尺寸进行转换
        x1 = int(x1 * self.original_size[0] / self.scaled_width)
        y1 = int(y1 * self.original_size[1] / self.scaled_height)
        x2 = int(x2 * self.original_size[0] / self.scaled_width)
        y2 = int(y2 * self.original_size[1] / self.scaled_height)
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, self.original_size[0]))
        y1 = max(0, min(y1, self.original_size[1]))
        x2 = max(0, min(x2, self.original_size[0]))
        y2 = max(0, min(y2, self.original_size[1]))
        
        # 如果选择的区域太小（比如只是点击了一下），使用全图
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            self.roi = (0, 0, self.original_size[0], self.original_size[1])
        else:
            # 转换为图像坐标
            self.roi = (x1, y1, x2, y2)
    
    def confirm(self):
        # 如果没有选择ROI，使用全图
        if not self.roi:
            # 直接使用原始图像的尺寸，不受缩放影响
            self.roi = (0, 0, self.original_size[0], self.original_size[1])
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedImageBlender(root)
    root.mainloop()