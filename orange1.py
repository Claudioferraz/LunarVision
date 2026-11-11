# ============================================================
#  main.py — Sensor de Linha com Câmera + Envio de Bits via UDP
#  Versão corrigida:
#   - Ponteiro na PONTA do indicador (landmark 8 + extrapolação)
#   - Suavização adaptativa (sem atraso quando move, sem tremer parado)
#   - Raios menores + confirmação de 2 frames (toque mais preciso)
#   - Teclas: +/- sensibilidade, R reseta linha, ESC sai
# ============================================================

import faulthandler
faulthandler.enable()

import sys
import traceback

try:
    import cv2
    print("[OK] opencv importado:", cv2.__version__)
except Exception as e:
    print("[ERRO] Falha ao importar OpenCV:", e)
    print("Rode: pip install opencv-python")
    input("ENTER para sair...")
    sys.exit(1)

try:
    import numpy as np
    print("[OK] numpy importado:", np.__version__)
except Exception as e:
    print("[ERRO] Falha ao importar numpy:", e)
    print("Rode: pip install numpy")
    input("ENTER para sair...")
    sys.exit(1)

try:
    import socket
    import tkinter as tk
    from tkinter import ttk, messagebox
    print("[OK] socket + tkinter importados")
except Exception as e:
    print("[ERRO] Falha ao importar tkinter:", e)
    input("ENTER para sair...")
    sys.exit(1)

# ---------- MediaPipe (opcional) ----------
MEDIAPIPE_OK = False
mp = None
try:
    import mediapipe as _mp
    if hasattr(_mp, "solutions") and hasattr(_mp.solutions, "hands"):
        mp = _mp
        MEDIAPIPE_OK = True
        print("[OK] MediaPipe (API solutions) disponível")
    else:
        print("[AVISO] MediaPipe instalado mas SEM 'solutions' (versão nova).")
        print("        Rode: pip install mediapipe==0.10.21")
        print("        Usando fallback por cor de pele.")
except ImportError:
    print("[AVISO] MediaPipe não instalado. Usando fallback por cor de pele.")
    print("        Para instalar: pip install mediapipe==0.10.21")
except Exception as e:
    print("[AVISO] Erro ao importar MediaPipe:", e)


# ============================================================
#  TELA DE CONFIGURAÇÃO
# ============================================================
class TelaConfig:
    def __init__(self):
        self.camera_index = None
        self.ip_destino = None
        self.porta = None

        self.root = tk.Tk()
        self.root.title("Configuração - Sensor de Linha")
        self.root.geometry("460x440")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._fechar)

        self._montar_ui()
        self.root.mainloop()

    def _fechar(self):
        self.camera_index = None
        self.root.destroy()

    def _montar_ui(self):
        pad = {"padx": 15, "pady": 6}

        tk.Label(self.root, text="Configuração do Sensor de Linha",
                 font=("Arial", 14, "bold")).pack(pady=15)

        frame_cam = tk.LabelFrame(self.root, text="Câmera", padx=10, pady=10)
        frame_cam.pack(fill="x", **pad)

        tk.Label(frame_cam, text="Selecione a câmera:").pack(anchor="w")
        self.combo_camera = ttk.Combobox(frame_cam, state="readonly", width=42)
        self.combo_camera.pack(pady=4)

        tk.Button(frame_cam, text="🔄 Detectar câmeras novamente",
                  command=self.detectar_cameras).pack(pady=4)

        self.label_info_cam = tk.Label(frame_cam, text="", fg="gray", font=("Arial", 9))
        self.label_info_cam.pack()

        frame_rede = tk.LabelFrame(self.root, text="Rede", padx=10, pady=10)
        frame_rede.pack(fill="x", **pad)

        tk.Label(frame_rede, text="IP de destino:").pack(anchor="w")
        self.entry_ip = tk.Entry(frame_rede, width=32)
        self.entry_ip.insert(0, "192.168.1.100")
        self.entry_ip.pack(pady=4)

        tk.Label(frame_rede, text="Porta:").pack(anchor="w")
        self.entry_porta = tk.Entry(frame_rede, width=32)
        self.entry_porta.insert(0, "5005")
        self.entry_porta.pack(pady=4)

        if not MEDIAPIPE_OK:
            tk.Label(self.root,
                     text="⚠ MediaPipe sem 'solutions'.\nDetecção usará cor de pele (menos precisa).",
                     fg="red", font=("Arial", 9)).pack(pady=4)

        frame_btn = tk.Frame(self.root)
        frame_btn.pack(pady=15)

        tk.Button(frame_btn, text="▶ Iniciar", width=12, bg="#4CAF50",
                  fg="white", font=("Arial", 11, "bold"),
                  command=self.iniciar).pack(side="left", padx=5)
        tk.Button(frame_btn, text="✖ Sair", width=12,
                  command=self._fechar).pack(side="left", padx=5)

        self.detectar_cameras()

    def detectar_cameras(self):
        self.combo_camera.set("Detectando...")
        self.label_info_cam.config(text="")
        self.root.update()

        disponiveis = []
        for i in range(6):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        disponiveis.append((i, w, h))
                cap.release()
            except Exception as e:
                print(f"[DEBUG] Erro ao testar câmera {i}: {e}")

        if not disponiveis:
            self.combo_camera.set("Nenhuma câmera encontrada")
            self.combo_camera["values"] = []
            self.label_info_cam.config(text="Verifique o cabo USB / driver")
            messagebox.showwarning("Aviso",
                                   "Nenhuma câmera detectada.\n\n"
                                   "Tente:\n"
                                   "• Reconectar cabo USB\n"
                                   "• Fechar Zoom/Teams/OBS\n"
                                   "• Ver Gerenciador de Dispositivos")
            return

        valores = [f"Câmera {i}  ({w}x{h})" for (i, w, h) in disponiveis]
        self.combo_camera["values"] = valores
        self.combo_camera.current(0)
        self.label_info_cam.config(text=f"{len(disponiveis)} câmera(s) encontrada(s)")

    def iniciar(self):
        sel = self.combo_camera.get()
        if not sel.startswith("Câmera"):
            messagebox.showerror("Erro", "Selecione uma câmera válida.")
            return
        try:
            self.camera_index = int(sel.split()[1])
        except Exception:
            messagebox.showerror("Erro", "Não foi possível ler o índice da câmera.")
            return

        ip = self.entry_ip.get().strip()
        partes = ip.split(".")
        if len(partes) != 4 or not all(p.isdigit() and 0 <= int(p) <= 255 for p in partes):
            messagebox.showerror("Erro", "IP inválido.")
            return
        self.ip_destino = ip

        try:
            porta = int(self.entry_porta.get().strip())
            if not (1 <= porta <= 65535):
                raise ValueError
        except ValueError:
            messagebox.showerror("Erro", "Porta inválida (1-65535).")
            return
        self.porta = porta

        self.root.destroy()


# ============================================================
#  SENSOR DE LINHA
# ============================================================
class SensorLinha:
    # ---- Parâmetros (ajuste aqui) ----
    RAIO_ENTRADA = 15        # px: distância pra ATIVAR (bit 1)
    FATOR_SAIDA = 2.0        # saída = entrada * fator (histerese)
    CONFIRMACAO_FRAMES = 2   # frames seguidos pra confirmar mudança
    PERDA_MAO_FRAMES = 6     # frames sem mão pra zerar o bit
    PONTA_EXTRA = 0.25       # empurra o ponto pra frente da ponta (0 = landmark puro)
    ALPHA_MIN = 0.35         # suavização com dedo parado (menor = mais suave)
    ALPHA_MAX = 0.95         # suavização com dedo rápido (maior = mais responsivo)

    def __init__(self, camera_index, ip_destino, porta):
        self.camera_index = camera_index
        self.ip_destino = ip_destino
        self.porta = porta

        self.raio_entrada = self.RAIO_ENTRADA

        self.pontos = []
        self.linha_p1 = None
        self.linha_p2 = None
        self.bit_atual = 0

        self.pos_suave = None
        self.frames_sem_mao = 0
        self.candidato_bit = 0
        self.contador_candidato = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        if MEDIAPIPE_OK:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
                model_complexity=1,
            )
        else:
            self.hands = None

    @property
    def raio_saida(self):
        return self.raio_entrada * self.FATOR_SAIDA

    # ---------- Mouse ----------
    def clique_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pontos) >= 2:
                self.resetar_linha()
            self.pontos.append((x, y))
            print(f"[DEBUG] Ponto: ({x}, {y})")
            if len(self.pontos) == 2:
                self.linha_p1, self.linha_p2 = self.pontos[0], self.pontos[1]
                print(f"[DEBUG] Linha: {self.linha_p1} -> {self.linha_p2}")

    def resetar_linha(self):
        self.pontos = []
        self.linha_p1 = None
        self.linha_p2 = None
        self.bit_atual = 0
        self.candidato_bit = 0
        self.contador_candidato = 0

    @staticmethod
    def distancia_ponto_linha(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return float(np.hypot(px - x1, py - y1))
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        return float(np.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))

    # ---------- Detecção ----------
    def detectar_mao_mediapipe(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        resultado = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not resultado.multi_hand_landmarks:
            return None

        mao = resultado.multi_hand_landmarks[0]
        h, w, _ = frame_bgr.shape
        self.mp_draw.draw_landmarks(frame_bgr, mao, self.mp_hands.HAND_CONNECTIONS)

        # Landmark 8 = ponta do indicador | 7 = articulação logo antes da ponta
        tip = mao.landmark[8]
        dip = mao.landmark[7]
        tx, ty = tip.x * w, tip.y * h
        dx, dy = dip.x * w, dip.y * h

        # Empurra um pouco além do landmark, na direção do dedo (DIP -> TIP)
        px = tx + (tx - dx) * self.PONTA_EXTRA
        py = ty + (ty - dy) * self.PONTA_EXTRA
        return (px, py)

    def detectar_mao_cor(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 30, 40], dtype=np.uint8)
        upper = np.array([25, 255, 255], dtype=np.uint8)
        mascara = cv2.inRange(hsv, lower, upper)
        mascara = cv2.erode(mascara, None, iterations=2)
        mascara = cv2.dilate(mascara, None, iterations=2)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return None
        maior = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(maior) < 1500:
            return None
        # Ponto mais alto do contorno = ponta do dedo apontando pra cima
        idx = maior[:, 0, 1].argmin()
        return (float(maior[idx, 0, 0]), float(maior[idx, 0, 1]))

    def detectar_mao(self, frame_bgr):
        if self.hands is not None:
            return self.detectar_mao_mediapipe(frame_bgr)
        return self.detectar_mao_cor(frame_bgr)

    def suavizar(self, ponto):
        """Filtro adaptativo: parado = suave, rápido = responsivo."""
        if ponto is None:
            return None
        novo = np.array(ponto, dtype=float)
        if self.pos_suave is None:
            self.pos_suave = novo
        else:
            vel = float(np.linalg.norm(novo - self.pos_suave))
            alpha = self.ALPHA_MIN + (self.ALPHA_MAX - self.ALPHA_MIN) * min(vel / 40.0, 1.0)
            self.pos_suave = self.pos_suave + alpha * (novo - self.pos_suave)
        return (int(round(self.pos_suave[0])), int(round(self.pos_suave[1])))

    # ---------- Histerese + confirmação ----------
    def atualizar_bit(self, dist):
        if dist is None:
            self.frames_sem_mao += 1
            if self.frames_sem_mao >= self.PERDA_MAO_FRAMES:
                self.bit_atual = 0
                self.candidato_bit = 0
                self.contador_candidato = 0
                self.pos_suave = None
            return

        self.frames_sem_mao = 0

        if self.bit_atual == 0:
            novo = 1 if dist <= self.raio_entrada else 0
        else:
            novo = 0 if dist > self.raio_saida else 1

        if novo == self.bit_atual:
            self.candidato_bit = novo
            self.contador_candidato = 0
            return

        if novo == self.candidato_bit:
            self.contador_candidato += 1
        else:
            self.candidato_bit = novo
            self.contador_candidato = 1

        if self.contador_candidato >= self.CONFIRMACAO_FRAMES:
            self.bit_atual = self.candidato_bit
            self.contador_candidato = 0

    def enviar_bit(self, bit):
        try:
            self.sock.sendto(str(bit).encode(), (self.ip_destino, self.porta))
        except Exception as e:
            print("[ERRO] enviar_bit:", e)

    # ---------- Loop ----------
    def rodar(self):
        print(f"[DEBUG] Abrindo câmera {self.camera_index} com DirectShow...")
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # menos atraso

        if not cap.isOpened():
            print(f"[ERRO] Câmera {self.camera_index} não abriu.")
            input("ENTER para sair...")
            return

        ret, teste = cap.read()
        if not ret or teste is None:
            print("[ERRO] Câmera abriu mas não retorna frames.")
            cap.release()
            input("ENTER para sair...")
            return

        print(f"[OK] Câmera {self.camera_index} aberta.")

        janela = "Sensor de Linha"
        cv2.namedWindow(janela)
        cv2.setMouseCallback(janela, self.clique_mouse)

        modo = "MediaPipe" if self.hands is not None else "Cor de pele"
        print(f"[OK] Modo detecção: {modo}")
        print(f"[OK] Enviando para {self.ip_destino}:{self.porta}")
        print("Clique 2 pontos. +/- sensibilidade | R reseta linha | ESC sai.")

        bit_anterior = 0
        falhas = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                falhas += 1
                if falhas > 30:
                    print("[AVISO] Câmera parou de enviar frames.")
                    break
                continue
            falhas = 0
            frame = cv2.flip(frame, 1)

            mao_raw = self.detectar_mao(frame)
            mao = self.suavizar(mao_raw)

            dist = None
            if mao is not None and self.linha_p1 and self.linha_p2:
                dist = self.distancia_ponto_linha(
                    mao[0], mao[1], *self.linha_p1, *self.linha_p2)

            self.atualizar_bit(dist)

            if self.bit_atual != bit_anterior:
                self.enviar_bit(self.bit_atual)
                print(f"[BIT] -> {self.bit_atual}")
                bit_anterior = self.bit_atual

            # ---- Desenho ----
            if self.linha_p1 and self.linha_p2:
                overlay = frame.copy()
                cv2.line(overlay, self.linha_p1, self.linha_p2,
                         (0, 100, 255), int(self.raio_saida * 2))
                cv2.line(overlay, self.linha_p1, self.linha_p2,
                         (0, 255, 0), int(self.raio_entrada * 2))
                frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
                cv2.line(frame, self.linha_p1, self.linha_p2, (0, 255, 255), 2)

            for p in self.pontos:
                cv2.circle(frame, p, 6, (255, 0, 0), -1)

            if mao is not None:
                cor_ponto = (0, 255, 0) if self.bit_atual == 1 else (0, 255, 255)
                cv2.circle(frame, mao, 7, cor_ponto, -1)
                cv2.circle(frame, mao, 9, (0, 0, 0), 1)
                cv2.putText(frame,
                            f"dist={dist:.0f}px" if dist is not None else "dist=--",
                            (15, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cor = (0, 255, 0) if self.bit_atual == 1 else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 0, 0), -1)
            cv2.putText(frame, f"BIT: {self.bit_atual}", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor, 2)
            cv2.putText(frame, f"raio={self.raio_entrada:.0f}px (+/-)",
                        (170, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(frame, f"{self.ip_destino}:{self.porta}",
                        (frame.shape[1] - 240, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            if not (self.linha_p1 and self.linha_p2):
                cv2.putText(frame, "Clique 2 pontos para criar a linha",
                            (int(frame.shape[1] / 2) - 220, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(janela, frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:
                break
            elif tecla in (ord('+'), ord('=')):
                self.raio_entrada = min(80, self.raio_entrada + 2)
            elif tecla in (ord('-'), ord('_')):
                self.raio_entrada = max(4, self.raio_entrada - 2)
            elif tecla in (ord('r'), ord('R')):
                self.resetar_linha()

        cap.release()
        cv2.destroyAllWindows()
        self.sock.close()
        if self.hands is not None:
            self.hands.close()


# ============================================================
#  PONTO DE ENTRADA
# ============================================================
if __name__ == "__main__":
    try:
        print("=" * 55)
        print("  SENSOR DE LINHA — iniciando")
        print("=" * 55)

        cfg = TelaConfig()

        if cfg.camera_index is None:
            print("[INFO] Configuração cancelada pelo usuário.")
        else:
            print(f"[INFO] Câmera escolhida: {cfg.camera_index}")
            print(f"[INFO] Destino: {cfg.ip_destino}:{cfg.porta}")
            app = SensorLinha(cfg.camera_index, cfg.ip_destino, cfg.porta)
            app.rodar()

        print("[INFO] Programa encerrado.")
    except Exception:
        print("\n[ERRO FATAL] Exceção não tratada:")
        traceback.print_exc()
        try:
            input("\nPressione ENTER para fechar...")
        except Exception:
            pass