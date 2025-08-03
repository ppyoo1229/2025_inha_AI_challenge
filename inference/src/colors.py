'''
이미지 컬러 추출  -> 프롬프트 강화용 -> HSV
'''
import numpy as np
from sklearn.cluster import KMeans
import webcolors

# from src.colors import (
#     color_words,
#     extract_dominant_colors,
#     print_dominant_colors_with_names,
#     closest_color_name
# )

# --- 이미지에서 주요 RGB 컬러 topk개 추출 --- 
def extract_dominant_colors(image, topk=3):
    img = image.resize((32, 32)).convert('RGB')
    arr = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=topk, n_init='auto').fit(arr)
    colors = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in colors]

# --- 주어진 RGB 튜플에 가장 가까운 CSS3 컬러명 매핑 ---
def print_dominant_colors_with_names(image_pil):
    dominant_colors = extract_dominant_colors(image_pil, topk=3)
    color_names = []
    for rgb_tuple in dominant_colors:
        try:
            name = webcolors.rgb_to_name(rgb_tuple, spec='css3')
        except Exception:
            # 최신 webcolors에서는 CSS3_NAMES_TO_HEX가 있는 경우도, 없는 경우도 있음
            try:
                names_map = webcolors.CSS3_NAMES_TO_HEX
            except AttributeError:
                names_map = {
                    # fallback 기본값 (필요시 여기에 140 CSS3 이름 추가)
                    'white': '#ffffff', 'black': '#000000', 'red': '#ff0000',
                    'blue': '#0000ff', 'green': '#008000', 'yellow': '#ffff00', 'orange': '#ffa500', 'pink': '#ffc0cb',
                    'purple': '#800080', 'brown': '#a52a2a', 'gray': '#808080', 'grey': '#808080'
                }
            min_dist = float("inf")
            closest_name = None
            for name, hex_code in names_map.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
                dist = (r_c - rgb_tuple[0]) ** 2 + (g_c - rgb_tuple[1]) ** 2 + (b_c - rgb_tuple[2]) ** 2
                if dist < min_dist:
                    min_dist = dist
                    closest_name = name
            name = closest_name
        color_names.append(name)
    print(f"[dominant RGB] {dominant_colors} | [color name] {color_names}")
    return color_names

