import cv2
import numpy as np
import os

# Configurações
LIMIAR_FILL = 0.40  # porcentagem mínima de preenchimento para considerar marcada
N_COLS = 3
QUESTOES_POR_COL = 20
_K3 = np.ones((3, 3), np.uint8)

def reorder(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos em sentido horário (TL, TR, BL, BR)."""
    pts = pts.reshape((4, 2)).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    add = pts.sum(1)
    diff = np.diff(pts, axis=1)
    ordered[0] = pts[np.argmin(add)]
    ordered[3] = pts[np.argmax(add)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[2] = pts[np.argmax(diff)]
    return ordered
if name == "main":
    respostas = testar_imagem_diretamente("caminho/para/sua_imagem.jpg")
    for q, r in respostas:
        print(f"Questão {q:02d}: {r}")