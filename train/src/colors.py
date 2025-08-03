def extract_dominant_colors(image, topk=3):
    img = image.resize((32,32)).convert('RGB')
    arr = np.array(img).reshape(-1,3)
    kmeans = KMeans(n_clusters=topk, n_init='auto').fit(arr)
    colors = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in colors]

def rgb_to_simple_color_name(rgb_tuple):
    min_dist = float("inf")
    closest_name = None
    for name, hex_code in webcolors.CSS3_NAMES_TO_HEX.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
        dist = (r_c - rgb_tuple[0])**2 + (g_c - rgb_tuple[1])**2 + (b_c - rgb_tuple[2])**2
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name