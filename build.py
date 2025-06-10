import PyInstaller.__main__
import os

# 确保当前目录是脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 定义图标文件路径（如果有的话）
# icon_path = 'icon.ico'  # 如果有图标文件，取消注释这行

# PyInstaller 参数
params = [
    '图像仿射变换小工具.py',  # 主程序文件
    '--name=图像仿射变换工具',  # 生成的exe名称
    '--windowed',  # 使用GUI模式
    '--noconfirm',  # 覆盖输出目录
    '--clean',  # 清理临时文件
    '--add-data=config.json;.',  # 添加配置文件
    # f'--icon={icon_path}',  # 如果有图标文件，取消注释这行
    '--hidden-import=PIL._tkinter_finder',  # 添加隐藏导入
]

# 运行 PyInstaller
PyInstaller.__main__.run(params) 