import collections
import collections.abc
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_CONNECTOR

# ==========================================
# 0. 初始化与调色盘
# ==========================================
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

blank_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_layout)

# 颜色定义
C_TITLE = RGBColor(27, 79, 114)
C_DARK = RGBColor(44, 62, 80)
C_MUTED = RGBColor(127, 140, 141)

C_BLOCK_BORDER = RGBColor(41, 128, 185)
C_SPATIAL_BG = RGBColor(234, 250, 241)
C_SPATIAL_BD = RGBColor(39, 174, 96)
C_TEMP_BG = RGBColor(254, 249, 231)
C_TEMP_BD = RGBColor(230, 126, 34)
C_RES_BG = RGBColor(242, 244, 244)
C_RES_BD = RGBColor(149, 165, 166)

C_GAP = RGBColor(235, 222, 240)
C_GAP_BD = RGBColor(155, 89, 182)
C_FC = RGBColor(253, 235, 208)
C_FC_BD = RGBColor(220, 118, 51)
C_SOFTMAX = RGBColor(250, 219, 216)
C_SOFTMAX_BD = RGBColor(217, 136, 128)

C_NODE_RED = RGBColor(231, 76, 60)
C_NODE_GREEN = RGBColor(46, 204, 113)
C_NODE_BLUE = RGBColor(52, 152, 219)
C_NODE_PURP = RGBColor(155, 89, 182)

# ==========================================
# 绘图辅助函数
# ==========================================
def set_text(shape, text, size=9, bold=False, color=C_DARK, align=PP_ALIGN.CENTER):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = "Arial"
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.05)
    tf.margin_bottom = Inches(0.05)

def add_box(slide, x, y, w, h, fill=None, border=None, dash=False, round_corner=False, is_oval=False, is_cube=False):
    # 底层数字映射：14=立方体, 9=圆, 5=圆角矩形, 1=直角矩形
    if is_cube:
        stype = 14
    elif is_oval:
        stype = 9 
    else:
        stype = 5 if round_corner else 1 
        
    shape = slide.shapes.add_shape(stype, Inches(x), Inches(y), Inches(w), Inches(h))
    
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
        
    if border:
        shape.line.color.rgb = border
        shape.line.width = Pt(1.5)
        if dash: shape.line.dash_style = 2
    else:
        shape.line.fill.background()
    return shape

def add_arrow(slide, sx, sy, ex, ey, color=C_DARK, width=1.5):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(sx), Inches(sy), Inches(ex), Inches(ey))
    conn.line.fill.solid()
    conn.line.color.rgb = color
    conn.line.width = Pt(width)
    conn.line.end_arrowhead = 5 # 5 = Stealth Arrow, 在PPT里显示更明显
    return conn

def draw_skeleton(slide, cx, cy, scale=0.4):
    """绘制标准人体骨架连线图"""
    # 骨架点位映射 (x, y)
    nodes = {
        'head': (0, -0.6), 'neck': (0, -0.3),
        'r_sh': (0.2, -0.3), 'l_sh': (-0.2, -0.3),
        'r_elb': (0.35, -0.05), 'l_elb': (-0.35, -0.05),
        'r_wri': (0.45, 0.2), 'l_wri': (-0.45, 0.2),
        'spine': (0, 0.15),
        'r_hip': (0.15, 0.4), 'l_hip': (-0.15, 0.4),
        'r_knee': (0.15, 0.8), 'l_knee': (-0.15, 0.8),
        'r_ank': (0.15, 1.2), 'l_ank': (-0.15, 1.2)
    }
    limbs = [('head','neck'), ('neck','r_sh'), ('neck','l_sh'), ('r_sh','r_elb'), 
             ('r_elb','r_wri'), ('l_sh','l_elb'), ('l_elb','l_wri'), ('neck','spine'),
             ('spine','r_hip'), ('spine','l_hip'), ('r_hip','r_knee'), ('r_knee','r_ank'),
             ('l_hip','l_knee'), ('l_knee','l_ank')]
    
    # 画连线
    for s, e in limbs:
        x1, y1 = nodes[s]; x2, y2 = nodes[e]
        conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(cx+x1*scale), Inches(cy+y1*scale), Inches(cx+x2*scale), Inches(cy+y2*scale))
        conn.line.color.rgb = RGBColor(52, 73, 94)
        conn.line.width = Pt(1.5)
    # 画节点
    for name, (nx, ny) in nodes.items():
        shape = slide.shapes.add_shape(9, Inches(cx+nx*scale-0.03), Inches(cy+ny*scale-0.03), Inches(0.06), Inches(0.06))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(52, 152, 219)

# ==========================================
# 1. 左侧：输入序列与维度
# ==========================================
tb = add_box(slide, 0.2, 0.2, 3.2, 0.8)
set_text(tb, "Overall ST-GCN Architecture\n(Skeleton-Based Action Recognition)", 13, True, C_TITLE, PP_ALIGN.LEFT)

set_text(add_box(slide, 0.2, 1.0, 2.9, 0.4), "Skeleton Sequence\n(N x C x T x V Input)", 10, True)

# 画三个收缩比例的独立骨架
add_box(slide, 0.2, 1.4, 3.0, 1.6, border=C_RES_BD)
draw_skeleton(slide, 0.7, 1.8, scale=0.7)
draw_skeleton(slide, 1.7, 1.8, scale=0.7)
draw_skeleton(slide, 2.7, 1.8, scale=0.7)

set_text(add_box(slide, 0.5, 2.75, 0.5, 0.2), "t = 1", 9)
set_text(add_box(slide, 1.5, 2.75, 0.5, 0.2), "t = 2", 9)
set_text(add_box(slide, 2.5, 2.75, 0.5, 0.2), "t = T", 9)
set_text(add_box(slide, 2.1, 1.8, 0.3, 0.3), "...", 14, True)

# 使用 PPT 原生的 3D 立方体生成 Feature Tensor
add_box(slide, 0.5, 3.2, 2.4, 1.2, fill=RGBColor(214, 234, 248), border=C_BLOCK_BORDER, is_cube=True)
set_text(add_box(slide, 0.2, 4.4, 3.0, 0.4), "Feature Tensor Map (C x T x V)", 10, True)

lg_box = add_box(slide, 0.2, 4.8, 3.0, 1.1, border=C_RES_BD)
set_text(lg_box, "N : Batch Size\nC : Channel (Feature Dimension)\nT : Temporal Length (Frames)\nV : Number of Joints (Vertices)", 9, align=PP_ALIGN.LEFT)

ex_box = add_box(slide, 0.2, 6.0, 3.0, 1.1, border=C_TEMP_BD, dash=True)
set_text(ex_box, "Example (NTU RGB+D)\nN x 3 x 300 x 25\n(x, y, z coordinates)\nC = 3, T = 300, V = 25", 9, align=PP_ALIGN.LEFT)

# ==========================================
# 2. 中间：ST-GCN 主干网络
# ==========================================
norm = add_box(slide, 3.5, 0.6, 2.5, 0.4, border=C_BLOCK_BORDER, round_corner=True)
set_text(norm, "Data Normalization\n(Batch Normalization)", 9.5, True)
add_arrow(slide, 4.75, 1.0, 4.75, 1.2)

add_box(slide, 3.4, 1.1, 2.7, 4.8, border=C_RES_BD, dash=True, round_corner=True)

def draw_block(slide, x, y, title):
    add_box(slide, x, y, 2.5, 1.2, border=C_BLOCK_BORDER, round_corner=True)
    set_text(add_box(slide, x+0.1, y+0.05, 2.3, 0.2), title, 9, True, C_BLOCK_BORDER)
    s_gcn = add_box(slide, x+0.15, y+0.3, 2.2, 0.23, fill=C_SPATIAL_BG, border=C_SPATIAL_BD, round_corner=True)
    set_text(s_gcn, "Spatial GCN", 8, True)
    t_conv = add_box(slide, x+0.15, y+0.58, 2.2, 0.23, fill=C_TEMP_BG, border=C_TEMP_BD, round_corner=True)
    set_text(t_conv, "Temporal Conv (TCN)", 8, True)
    res = add_box(slide, x+0.15, y+0.86, 2.2, 0.23, fill=C_RES_BG, border=C_RES_BD, round_corner=True)
    set_text(res, "Residual Connection", 8, True)

draw_block(slide, 3.5, 1.2, "ST-GCN Block 1")
add_arrow(slide, 4.75, 2.4, 4.75, 2.6)
set_text(add_box(slide, 4.65, 2.6, 0.2, 0.2, border=C_DARK, is_oval=True), "+", 9, True)
add_arrow(slide, 4.75, 2.8, 4.75, 3.0)

draw_block(slide, 3.5, 3.0, "ST-GCN Block 2")
add_arrow(slide, 4.75, 4.2, 4.75, 4.35)
set_text(add_box(slide, 4.65, 4.35, 0.2, 0.2, border=C_DARK, is_oval=True), "+", 9, True)

set_text(add_box(slide, 4.5, 4.55, 0.5, 0.2), "...", 14, True)
add_arrow(slide, 4.75, 4.75, 4.75, 4.9)

draw_block(slide, 3.5, 4.9, "ST-GCN Block L (e.g., Block 9)")
add_arrow(slide, 4.75, 6.1, 4.75, 6.25)

gap = add_box(slide, 3.5, 6.25, 2.5, 0.3, fill=C_GAP, border=C_GAP_BD, round_corner=True)
set_text(gap, "Global Average Pooling (GAP)", 9, True)
add_arrow(slide, 4.75, 6.55, 4.75, 6.65)

fc = add_box(slide, 3.5, 6.65, 2.5, 0.3, fill=C_FC, border=C_FC_BD, round_corner=True)
set_text(fc, "Fully Connected Layer", 9, True)
add_arrow(slide, 4.75, 6.95, 4.75, 7.05)

sm = add_box(slide, 3.5, 7.05, 2.5, 0.35, fill=C_SOFTMAX, border=C_SOFTMAX_BD, round_corner=True)
set_text(sm, "Softmax Classifier (Action Classes)", 9, True)

# ==========================================
# 3. 右上：Spatial GCN
# ==========================================
add_box(slide, 6.3, 0.2, 6.8, 3.4, border=C_SPATIAL_BD, round_corner=True)
set_text(add_box(slide, 6.4, 0.25, 4.0, 0.3), "Spatial GCN (Spatial Graph Convolution)", 11, True, C_SPATIAL_BD, PP_ALIGN.LEFT)

set_text(add_box(slide, 6.4, 0.6, 1.4, 0.5), "Input (One Frame)\nX ∈ R^(C_in × V)", 8, True)
draw_skeleton(slide, 7.1, 1.8, scale=0.7)
add_arrow(slide, 7.7, 1.8, 8.1, 1.8)

set_text(add_box(slide, 8.2, 0.6, 1.8, 0.5), "Human Skeleton Graph\nAdjacency Matrix A", 8, True)
grid_top, grid_left, g_size = 1.3, 8.4, 0.18
# 块对角线矩阵渲染，模拟真实特征
for r in range(8):
    for c in range(8):
        # 让 2x2 作为一个小块变绿，模拟局部邻接
        f_color = C_SPATIAL_BD if (r//2 == c//2) or r==c else RGBColor(255,255,255)
        add_box(slide, grid_left + c*g_size, grid_top + r*g_size, g_size, g_size, fill=f_color, border=C_RES_BD)
set_text(add_box(slide, 8.2, 2.8, 1.8, 0.3), "A_ij = 1 if neighbor or self", 8)
add_arrow(slide, 10.0, 1.8, 10.4, 1.8)

set_text(add_box(slide, 10.4, 0.6, 1.3, 0.3), "Graph Convolution", 8, True)
ccx, ccy = 11.2, 1.8
cn = add_box(slide, ccx-0.15, ccy-0.15, 0.3, 0.3, fill=C_NODE_RED, border=C_NODE_RED, is_oval=True)
set_text(cn, "i", 10, True, RGBColor(255,255,255))
set_text(add_box(slide, 10.9, 1.3, 1.0, 0.2), "Center Joint i", 8, True, C_NODE_RED)

neighbors = [(0.5, 0), (-0.4, 0.4), (-0.3, -0.4), (0.3, -0.5)]
for idx, (nx, ny) in enumerate(neighbors):
    nb = add_box(slide, ccx+nx-0.1, ccy+ny-0.1, 0.2, 0.2, fill=C_NODE_GREEN, border=C_NODE_GREEN, is_oval=True)
    set_text(nb, f"j{idx+1}", 7)
    # 箭头指向中心节点
    add_arrow(slide, ccx+nx*0.8, ccy+ny*0.8, ccx+nx*0.3, ccy+ny*0.3, color=C_DARK, width=1.0)

add_arrow(slide, 11.8, 1.8, 12.2, 1.8)

set_text(add_box(slide, 12.2, 0.6, 0.9, 0.4), "Output\nX' ∈ R^(C_out × V)", 8, True)
for i in range(4):
    add_box(slide, 12.5, 1.3 + i*0.3, 0.2, 0.2, fill=C_NODE_BLUE, border=C_NODE_BLUE, is_oval=True)

set_text(add_box(slide, 7.5, 3.1, 5.4, 0.4, border=C_SPATIAL_BD, round_corner=True), "在同一帧内，根据骨架拓扑结构(邻接矩阵A)聚合每个关节的邻居信息，学习空间依赖关系。", 9, False, C_SPATIAL_BD)

# ==========================================
# 4. 右下：Temporal Conv
# ==========================================
add_box(slide, 6.3, 3.8, 6.8, 3.0, border=C_TEMP_BD, round_corner=True)
set_text(add_box(slide, 6.4, 3.85, 4.0, 0.3), "Temporal Conv (Temporal Convolution)", 11, True, C_TEMP_BD, PP_ALIGN.LEFT)

set_text(add_box(slide, 6.4, 4.3, 1.5, 0.4), "Input (One Joint)\nX ∈ R^(C_in × T)", 8, True)
dot_colors = [C_NODE_GREEN, C_NODE_BLUE, C_NODE_PURP]
for r in range(3):
    for c in range(5):
        add_box(slide, 6.5+c*0.25, 4.9+r*0.3, 0.15, 0.15, fill=dot_colors[r], border=dot_colors[r], is_oval=True)
    set_text(add_box(slide, 6.5+5*0.25, 4.85+r*0.3, 0.3, 0.2), "...", 10, True)

add_arrow(slide, 8.0, 5.15, 8.4, 5.15)

set_text(add_box(slide, 8.3, 4.3, 2.0, 0.4), "1D Convolution\n(On Time Dimension)", 8, True)
for r in range(3):
    for c in range(5):
        add_box(slide, 8.5+c*0.25, 4.9+r*0.3, 0.15, 0.15, fill=dot_colors[r], border=dot_colors[r], is_oval=True)

# 虚线滑窗框住中间三列 (c=1,2,3)
add_box(slide, 8.5 + 1*0.25 - 0.05, 4.8, 3*0.25 + 0.05, 0.9, border=C_TEMP_BD, dash=True, round_corner=True)
set_text(add_box(slide, 8.8, 5.8, 1.0, 0.3, fill=C_TEMP_BG, border=C_TEMP_BD), "Kernel Size = K_t", 8)

add_arrow(slide, 10.0, 5.15, 10.4, 5.15)

set_text(add_box(slide, 10.5, 4.3, 1.5, 0.4), "Output\nX' ∈ R^(C_out × T)", 8, True)
for r in range(3):
    for c in range(4):
        add_box(slide, 10.6+c*0.25, 4.9+r*0.3, 0.15, 0.15, fill=C_TEMP_BD, border=C_TEMP_BD, is_oval=True)

set_text(add_box(slide, 7.5, 6.2, 5.4, 0.4, border=C_TEMP_BD, round_corner=True), "在同一关节上，沿时间维度进行 1D 卷积，学习动作时间变化的动态信息。", 9, False, C_TEMP_BD)

# ==========================================
# 5. 跨模块虚线连接指示
# ==========================================
conn1 = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(5.85), Inches(1.6), Inches(6.3), Inches(1.0))
conn1.line.color.rgb = C_SPATIAL_BD
conn1.line.width = Pt(1.5)
conn1.line.dash_style = 2

conn2 = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(5.85), Inches(1.9), Inches(6.3), Inches(4.5))
conn2.line.color.rgb = C_TEMP_BD
conn2.line.width = Pt(1.5)
conn2.line.dash_style = 2

# ==========================================
# 保存
# ==========================================
prs.save("ST_GCN_Architecture_Final.pptx")
print("Presentation successfully saved as ST_GCN_Architecture_Final.pptx")