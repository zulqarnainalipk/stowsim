"""Draw the processing-flow diagram (English) with Pillow."""
from PIL import Image, ImageDraw, ImageFont
import sys

W, H = 1650, 870
S = 2  # supersampling for smooth edges
img = Image.new('RGB', (W * S, H * S), 'white')
d = ImageDraw.Draw(img)
TEAL = (21, 94, 130)
BLUE = (0, 112, 192)
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)

def font(size):
    for name in ('segoeui.ttf', 'arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, int(size * S))
        except OSError:
            continue
    return ImageFont.load_default()

F = font(27)
FS = font(25)
FH = font(28)

def box(x0, y0, x1, y1, text, color=TEAL):
    d.rectangle([x0 * S, y0 * S, x1 * S, y1 * S], fill=color)
    center_text((x0 + x1) / 2, (y0 + y1) / 2, text)

def center_text(cx, cy, text, fill=WHITE, f=None):
    f = f or F
    lines = text.split('\n')
    lh = f.size / S * 1.25
    y = cy - lh * len(lines) / 2
    for ln in lines:
        w = d.textlength(ln, font=f) / S
        d.text(((cx - w / 2) * S, y * S), ln, fill=fill, font=f)
        y += lh

def diamond(cx, cy, w, h, text):
    pts = [(cx * S, (cy - h / 2) * S), ((cx + w / 2) * S, cy * S), (cx * S, (cy + h / 2) * S), ((cx - w / 2) * S, cy * S)]
    d.polygon(pts, fill=TEAL)
    center_text(cx, cy, text)

def line(points, arrow=True, color=TEAL, width=3):
    pts = [(x * S, y * S) for x, y in points]
    d.line(pts, fill=color, width=width * S, joint='curve')
    if arrow:
        (x0, y0), (x1, y1) = points[-2], points[-1]
        import math
        a = math.atan2(y1 - y0, x1 - x0)
        L, Wd = 13, 7
        p1 = (x1, y1)
        p2 = (x1 - L * math.cos(a) + Wd * math.sin(a), y1 - L * math.sin(a) - Wd * math.cos(a))
        p3 = (x1 - L * math.cos(a) - Wd * math.sin(a), y1 - L * math.sin(a) + Wd * math.cos(a))
        d.polygon([(p[0] * S, p[1] * S) for p in (p1, p2, p3)], fill=color)

def label(x, y, text):
    d.text((x * S, y * S), text, fill=BLACK, font=FS)

d.text((20 * S, 8 * S), 'Agent (your code)', fill=BLACK, font=FH)
d.text((800 * S, 8 * S), 'Environment (loading simulator)', fill=BLACK, font=FH)

box(1040, 55, 1196, 100, 'Start')
line([(1118, 100), (1118, 130)], arrow=False)
diamond(1118, 197, 390, 140, 'Offline\noptimisation?')
# yes branch
line([(923, 198), (743, 198)])
label(820, 157, 'Yes')
box(467, 158, 741, 238, 'Pass the bag and\ncontainer information')
line([(467, 198), (289, 198)])
box(92, 158, 287, 238, 'Run offline\noptimisation', BLUE)
line([(189, 238), (189, 373), (443, 373)])
box(444, 330, 764, 417, 'Pass the optimised\nloading order')
line([(764, 373), (958, 372)])
# no branch
line([(1118, 267), (1118, 332)])
label(1148, 272, 'No')
box(959, 333, 1278, 410, 'Set up the bag stream\nand the containers')
line([(1118, 410), (1118, 480)])
box(905, 481, 1333, 573, 'Generate the observation:\ncontainer contents, pooled bags, ...')
box(1406, 489, 1636, 566, 'Refill the pool\nwith bags')
line([(1406, 527), (1335, 527)])
line([(905, 527), (289, 527)])
box(92, 487, 287, 567, 'Run online\noptimisation', BLUE)
line([(189, 567), (189, 683), (411, 683)])
box(411, 628, 822, 738, 'Place the chosen bag in the\nchosen container, at the\nchosen position and orientation')
line([(822, 683), (898, 683)])
diamond(1123, 682, 450, 140, 'Bags left in\nthe stock?')
line([(1349, 682), (1522, 682), (1522, 568)])
label(1388, 640, 'Yes')
line([(1123, 752), (1123, 797)])
label(1135, 755, 'No')
box(1067, 797, 1181, 843, 'End')

img = img.resize((W, H), Image.LANCZOS)
img.save(sys.argv[1] if len(sys.argv) > 1 else 'flow.png')
print('saved')
