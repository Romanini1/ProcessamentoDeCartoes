def split_columns(img: np.ndarray, n_cols: int = 3, marg_frac: float = 0.03):
    h, w = img.shape[:2]
    m = int(w * marg_frac)
    cw = (w - 2 * m) // n_cols
    return [img[:, m + i * cw : m + (i + 1) * cw] if i < n_cols - 1 else img[:, m + i * cw : w - m] for i in range(n_cols)]
# -------------------------  Agrupamento em linhas -------------------------

def _square_contours(col_img: np.ndarray):
    gray = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, _K3, iterations=1)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sq = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 10 < w < 60 and 0.75 <= w / h <= 1.3:
            sq.append((y + h // 2, x, y, h))
    sq.sort(key=lambda s: s[0])
    if sq:
        avg_h = np.mean([h for _, _, _, h in sq])
        sq = [s for s in sq if s[3] < 1.5 * avg_h]
    return sq

def split_rows(col_img: np.ndarray, expected: int = 20):
    sq = _square_contours(col_img)
    if not sq:
        return []

    clusters = []
    for cy, _, y, h in sq:
        if not clusters or cy - clusters[-1][-1][0] > 15:
            clusters.append([(cy, y, h)])
        else:
            clusters[-1].append((cy, y, h))

    h_col = col_img.shape[0]
    clusters = [cl for cl in clusters if 0.06 * h_col < cl[0][0] < 0.94 * h_col]
    clusters.sort(key=lambda c: c[0][0])
    while len(clusters) > expected:
        clusters.pop(0)
        if len(clusters) > expected:
            clusters.pop()

    linhas = []
    for cl in clusters:
        top = min(y for _, y, _ in cl)
        bottom = max(y + h for _, y, h in cl)
        margin = int((bottom - top) * 0.10)
        linhas.append(col_img[top + margin : bottom, :])
    return linhas