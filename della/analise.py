def testar_imagem_diretamente(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    warp = detectar_e_retificar(img)
    if warp is None:
        raise RuntimeError("Falha na retificação da imagem.")

    cols = split_columns(warp)
    letras = list("ABCDE")
    questao = 1
    resultados = []

    for col in cols:
        linhas = split_rows(col)
        for linha in linhas:
            gray = cv2.cvtColor(linha, cv2.COLOR_BGR2GRAY)
            thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            quads = []
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if not (15 < w < 70 and 0.75 < w/h < 1.25):
                    continue
                dx, dy = int(w*0.2), int(h*0.2)
                core = thr[y+dy:y+h-dy, x+dx:x+w-dx]
                if core.size == 0:
                    continue
                fill = cv2.countNonZero(core) / core.size
                quads.append((x, fill))

            if len(quads) < 5:
                resultados.append((questao, "-"))
            else:
                quads.sort(key=lambda q: q[0])
                fills = [q[1] for q in quads[:5]]
                marcadas = [i for i, f in enumerate(fills) if f >= LIMIAR_FILL]
                if not marcadas:
                    resultados.append((questao, "-"))
                else:
                    idx_escolhido = max(marcadas, key=lambda i: fills[i])
                    resultados.append((questao, letras[idx_escolhido]))

            questao += 1

    return resultados