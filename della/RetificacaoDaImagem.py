def detectar_e_retificar(img: np.ndarray):
    def _localizar_triangulos(img: np.ndarray, frac: float = 0.18):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lim_x, lim_y = int(w * frac), int(h * frac)
        marcadores: dict[str, tuple[int, int, float]] = {}

        for c in cnts:
            if len(cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)) != 3:
                continue
            M = cv2.moments(c)
            if not M["m00"]:
                continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            area = cv2.contourArea(c)

            if cx < lim_x and cy < lim_y:
                k = "TL"
            elif cx > w - lim_x and cy < lim_y:
                k = "TR"
            elif cx < lim_x and cy > h - lim_y:
                k = "BL"
            elif cx > w - lim_x and cy > h - lim_y:
                k = "BR"
            else:
                continue

            if k not in marcadores or area > marcadores[k][2]:
                marcadores[k] = (cx, cy, area)

        return {k: (v[0], v[1]) for k, v in marcadores.items()}

    verts = _localizar_triangulos(img)
    if len(verts) != 4:
        print("⚠  Marcadores insuficientes – imagem ignorada (não retificada)")
        return None

    pts = reorder(np.array([verts[k] for k in ("TL", "TR", "BL", "BR")], dtype=np.float32))

    width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    height = int(max(np.linalg.norm(pts[0] - pts[2]), np.linalg.norm(pts[1] - pts[3])))

    dst = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (width, height))
