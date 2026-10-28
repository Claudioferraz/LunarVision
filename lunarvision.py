import time
import threading
import traceback
import urllib.request
from pathlib import Path
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
from PIL import Image, ImageTk


# --- Configurações ajustáveis -------------------------------------------------
DETECT_WIDTH = 640            # largura do frame enviado ao MediaPipe (menor = mais rápido)
DISPLAY_MAX = (920, 600)      # tamanho máximo do preview na janela
SMOOTH_ALPHA = 0.5            # suavização do cursor: 1.0 = sem suavizar, menor = mais suave (e mais lento)
DEADZONE_PX = 2               # ignora movimentos menores que isso (elimina o tremor parado)
HAND_SCALE_MIN = 0.03         # tamanho mínimo da mão (normalizado 0-1) para aceitar gestos
HAND_SCALE_MAX = 0.60         # tamanho máximo da mão (normalizado 0-1) para aceitar gestos
MAX_CAMERA_INDEX = 6          # quantos índices de webcam testar na busca
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_MIN_BYTES = 1_000_000


# --- Reconhecimento de gestos -------------------------------------------------
# Cada dedo é descrito pelo par (ponta, articulação PIP). Se a ponta está mais longe
# do pulso (landmark 0) que a articulação PIP, consideramos o dedo "esticado".
FINGER_LANDMARKS = {
    "thumb": (4, 2),
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}
FINGER_EXTENSION_MARGIN = 1.1


def _landmark_distance(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def finger_states(hand_landmarks):
    wrist = hand_landmarks[0]
    states = {}
    for name, (tip_idx, pip_idx) in FINGER_LANDMARKS.items():
        tip = hand_landmarks[tip_idx]
        pip = hand_landmarks[pip_idx]
        states[name] = _landmark_distance(tip, wrist) > _landmark_distance(pip, wrist) * FINGER_EXTENSION_MARGIN
    return states


def classify_gesture(hand_landmarks):
    """
    "clique"  -> indicador + médio esticados, anelar e mindinho dobrados (paz).
    "arraste" -> mão fechada (mantém o botão pressionado).
    "apontar" -> qualquer outro caso (só move o cursor).
    """
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    # Os landmarks do MediaPipe são normalizados (0 a 1), então a comparação
    # de tamanho também precisa ser normalizada (antes comparava com pixels).
    hand_scale = _landmark_distance(wrist, middle_mcp)
    if not (HAND_SCALE_MIN <= hand_scale <= HAND_SCALE_MAX):
        return "apontar"

    states = finger_states(hand_landmarks)

    if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
        return "clique"
    if not any(states.values()):
        return "arraste"
    return "apontar"


@dataclass
class Calibration:
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    ready: bool = False

    def set_points(self, p1, p2):
        self.x1, self.x2 = sorted((int(p1[0]), int(p2[0])))
        self.y1, self.y2 = sorted((int(p1[1]), int(p2[1])))
        self.ready = self.x2 - self.x1 > 20 and self.y2 - self.y1 > 20

    def to_percentages(self, x, y):
        """Posição da câmera -> X/Y de 0 a 100% dentro da lousa. None se estiver fora."""
        if not self.ready:
            return None
        if x < self.x1 or x > self.x2 or y < self.y1 or y > self.y2:
            return None
        percent_x = (x - self.x1) / (self.x2 - self.x1) * 100.0
        percent_y = (y - self.y1) / (self.y2 - self.y1) * 100.0
        return percent_x, percent_y

    @staticmethod
    def percentages_to_screen(percent_x, percent_y, screen_w, screen_h):
        return (
            int(min(max(percent_x / 100.0, 0.0), 1.0) * (screen_w - 1)),
            int(min(max(percent_y / 100.0, 0.0), 1.0) * (screen_h - 1)),
        )


class LunarVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LunarVision — Protótipo")
        self.root.geometry("960x740")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.cap = None
        self.camera_options = []
        self.camera_index = 0
        self.camera_backend = cv2.CAP_DSHOW
        self.running = False
        self.calibrating = False
        self.calibration_stage = 0
        self.points = []
        self.calibration = Calibration()
        self.cursor_enabled = False

        # Informações do último frame (usadas na calibração por clique)
        self.latest_shape = None      # (altura, largura) do frame da câmera
        self.display_size = (0, 0)    # tamanho real da imagem exibida no preview

        # Controle do loop e da busca de câmeras
        self._after_id = None
        self._scanning = False
        self._scan_result = None

        # Suavização do cursor
        self.smooth_pos = None        # posição suavizada (float)
        self.last_cursor = None       # última posição realmente enviada ao mouse

        # Estado do reconhecimento de gestos
        self.current_gesture = None
        self.pending_gesture = None
        self.pending_since = 0.0
        self.gesture_confirm_seconds = 0.08
        self.hold_gesture = None
        self.last_click_time = 0.0
        self.click_cooldown_seconds = 0.5
        self.click_rearm_seconds = 0.15   # tempo fora do "paz" antes de permitir novo clique
        self.click_armed = True
        self.clique_released_at = None

        self.screen_w, self.screen_h = pyautogui.size()
        self.video_timestamp_ms = 0
        self.hands = self.create_hand_tracker()

        self.build_ui()
        self.scan_cameras()

    # --- Interface -------------------------------------------------------------
    def build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Webcam:").pack(side="left", padx=(0, 4))
        self.camera_combo = ttk.Combobox(top, state="readonly", width=30)
        self.camera_combo.pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Atualizar", command=self.scan_cameras).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Conectar", command=self.connect_selected_camera).pack(side="left", padx=(0, 10))
        self.calibrate_button = ttk.Button(top, text="Calibrar", command=self.start_calibration)
        self.calibrate_button.pack(side="left", padx=(0, 8))
        ttk.Button(top, text="Encerrar", command=self.close).pack(side="right")

        self.status = tk.StringVar(value="Procurando webcam...")
        ttk.Label(self.root, textvariable=self.status, padding=(10, 0)).pack(anchor="w")
        info = ("Escolha uma webcam e pressione Conectar. Depois pressione Calibrar e clique nos dois cantos "
                "da área da lousa na imagem. O cursor segue o seu dedo indicador. Gestos: "
                "indicador = apontar; 'paz' = clique; mão fechada = arraste.")
        ttk.Label(self.root, text=info, wraplength=920, padding=10).pack(anchor="w")
        self.preview = ttk.Label(self.root)
        self.preview.pack(expand=True, fill="both", padx=10, pady=10)

    # --- Busca de webcams (em thread, para não travar a janela) ----------------
    def scan_cameras(self):
        if self._scanning:
            return
        self.stop_camera()
        self._scanning = True
        self._scan_result = None
        self.status.set("Procurando webcams com DirectShow e Media Foundation...")
        threading.Thread(target=self._scan_worker, daemon=True).start()
        self.root.after(100, self._poll_scan)

    def _scan_worker(self):
        found = []
        for index in range(MAX_CAMERA_INDEX):
            for backend, backend_name in ((cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation")):
                test_cap = cv2.VideoCapture(index, backend)
                try:
                    if test_cap.isOpened():
                        ok, frame = test_cap.read()
                        if ok and frame is not None and frame.size:
                            found.append((index, backend, f"Índice {index} — {backend_name}"))
                finally:
                    test_cap.release()
        self._scan_result = found

    def _poll_scan(self):
        if self._scan_result is None:
            self.root.after(100, self._poll_scan)
            return
        found = self._scan_result
        self._scan_result = None
        self._scanning = False
        self.camera_options = found
        self.camera_combo["values"] = [item[2] for item in found]
        if found:
            self.camera_combo.current(0)
            self.status.set(f"{len(found)} opção(ões) encontrada(s). Pressione Conectar.")
        else:
            self.camera_combo.set("")
            self.status.set("Nenhuma webcam encontrada. Verifique o driver e feche outros aplicativos que usam a câmera.")

    # --- Câmera ----------------------------------------------------------------
    def connect_selected_camera(self):
        selected = self.camera_combo.current()
        if selected < 0 or selected >= len(self.camera_options):
            messagebox.showwarning("Webcam", "Selecione uma webcam antes de conectar.")
            return
        self.camera_index, self.camera_backend, _ = self.camera_options[selected]
        self.stop_camera()
        self.start_camera(self.camera_index, self.camera_backend)

    def start_camera(self, index, backend):
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            self.status.set("Não foi possível abrir a webcam escolhida. Tente o outro backend.")
            self.cap.release()
            self.cap = None
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # menos atraso entre a mão e o cursor
        self.running = True
        self._reset_tracking()
        self.status.set("Webcam conectada. Pressione Calibrar.")
        self._schedule_next(0)

    def stop_camera(self):
        self.running = False
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self._release_hold()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _schedule_next(self, delay_ms):
        # Guarda o id para poder cancelar. Sem isso, reconectar a câmera deixava
        # dois loops de update_frame rodando ao mesmo tempo.
        self._after_id = self.root.after(delay_ms, self.update_frame)

    # --- Loop principal --------------------------------------------------------
    def update_frame(self):
        self._after_id = None
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.status.set("A webcam conectou, mas não entregou imagem.")
            self._schedule_next(50)
            return

        frame = cv2.flip(frame, 1)  # espelha (corrige o eixo X invertido)
        h, w = frame.shape[:2]
        self.latest_shape = (h, w)

        detection = self.detect_hand(frame)  # detecta no frame reduzido
        display = frame                       # flip já criou uma cópia: pode desenhar direto

        if self.calibrating:
            text = "Clique no CANTO SUPERIOR ESQUERDO" if self.calibration_stage == 0 else "Clique no CANTO INFERIOR DIREITO"
            cv2.putText(display, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        for index, point in enumerate(self.points):
            cv2.circle(display, point, 10, (0, 255, 0), -1)
            cv2.putText(display, str(index + 1), (point[0] + 12, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.calibration.ready:
            cv2.rectangle(display, (self.calibration.x1, self.calibration.y1),
                          (self.calibration.x2, self.calibration.y2), (255, 0, 0), 2)

        if detection is not None:
            landmarks, (tx, ty), raw_gesture = detection
            for lx, ly in landmarks:
                cv2.circle(display, (lx, ly), 2, (0, 255, 0), -1)
            cv2.circle(display, (tx, ty), 12, (0, 0, 255), 2)
            cv2.circle(display, (tx, ty), 4, (255, 255, 255), -1)

            now = time.monotonic()
            gesture = self._confirm_gesture(raw_gesture, now)
            percentages = self.calibration.to_percentages(tx, ty)

            if percentages is not None and self.cursor_enabled:
                raw_mapped = Calibration.percentages_to_screen(*percentages, self.screen_w, self.screen_h)
                target = self._smooth(raw_mapped)
                try:
                    cursor = self._move_cursor(target)
                    self._update_hold_gesture(gesture, cursor)
                    self._update_click_gesture(gesture, cursor, now)
                except Exception:
                    print("[LunarVision] Erro ao mover/clicar com pyautogui:")
                    traceback.print_exc()
                    self.status.set("Erro no pyautogui — veja o console.")
                cv2.putText(display, f"Lousa: X {percentages[0]:.1f}% | Y {percentages[1]:.1f}%", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                cv2.putText(display, f"Tela atual: {target[0]}, {target[1]}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                cv2.putText(display, f"Gesto: {gesture}", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
            else:
                # Mão fora da área da lousa (ou cursor desligado): solta o botão
                # para o arraste não ficar preso e reinicia a suavização.
                self._release_hold()
                self.smooth_pos = None
                self.last_cursor = None
                self._rearm_click()

            self.current_gesture = gesture
        else:
            self._release_hold()
            self._reset_tracking()

        self._render_preview(display)
        self._schedule_next(1)

    def _render_preview(self, display):
        h, w = display.shape[:2]
        scale = min(DISPLAY_MAX[0] / w, DISPLAY_MAX[1] / h, 1.0)
        dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(display, (dw, dh), interpolation=cv2.INTER_AREA) if scale < 1.0 else display
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.display_size = (dw, dh)
        self.preview.configure(image=photo)
        self.preview.image = photo

    # --- Detecção --------------------------------------------------------------
    def create_hand_tracker(self):
        model_path = Path(__file__).with_name("hand_landmarker.task")

        if model_path.exists() and model_path.stat().st_size < MODEL_MIN_BYTES:
            print(f"[LunarVision] hand_landmarker.task parece corrompido "
                  f"({model_path.stat().st_size} bytes). Apagando para baixar de novo.")
            model_path.unlink()

        if not model_path.exists():
            print("[LunarVision] Baixando o modelo de detecção da mão...")
            tmp_path = model_path.with_suffix(".tmp")
            try:
                urllib.request.urlretrieve(MODEL_URL, tmp_path)
                tmp_path.replace(model_path)  # só vira o arquivo final se o download terminou
            except Exception as exc:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"Não foi possível baixar hand_landmarker.task: {exc}") from exc
            if model_path.stat().st_size < MODEL_MIN_BYTES:
                model_path.unlink()
                raise RuntimeError("O modelo baixado está incompleto. Tente novamente.")

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.50,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.60,
        )
        print(f"[LunarVision] Modelo carregado de {model_path} ({model_path.stat().st_size} bytes).")
        return vision.HandLandmarker.create_from_options(options)

    def detect_hand(self, frame):
        """Retorna (lista de pontos em pixels, (x, y) do indicador em pixels, gesto) ou None."""
        h, w = frame.shape[:2]
        try:
            # Os landmarks são normalizados, então detectar num frame menor
            # dá o mesmo resultado, só que bem mais rápido.
            if w > DETECT_WIDTH:
                small = cv2.resize(frame, (DETECT_WIDTH, int(h * DETECT_WIDTH / w)), interpolation=cv2.INTER_AREA)
            else:
                small = frame
            rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamp real e sempre crescente (o MediaPipe exige isso no modo VIDEO).
            ts = int(time.monotonic() * 1000)
            if ts <= self.video_timestamp_ms:
                ts = self.video_timestamp_ms + 1
            self.video_timestamp_ms = ts

            result = self.hands.detect_for_video(mp_image, ts)
        except Exception:
            print("[LunarVision] Erro na detecção da mão:")
            traceback.print_exc()
            self.status.set("Erro na detecção da mão — veja o console.")
            return None

        if not result.hand_landmarks:
            self._log_no_hand_once()
            return None

        hand = result.hand_landmarks[0]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        tip = points[8]  # ponta do indicador
        return points, tip, classify_gesture(hand)

    def _log_no_hand_once(self):
        now = time.monotonic()
        if now - getattr(self, "_last_no_hand_log", 0) > 1.0:
            self._last_no_hand_log = now
            print("[LunarVision] Nenhuma mão detectada neste frame.")

    # --- Cursor e gestos -------------------------------------------------------
    def _reset_tracking(self):
        self.current_gesture = None
        self.pending_gesture = None
        self.smooth_pos = None
        self.last_cursor = None
        self._rearm_click()

    def _rearm_click(self):
        self.click_armed = True
        self.clique_released_at = None

    def _smooth(self, pos):
        """Média móvel exponencial: reduz o tremor sem o atraso de uma média grande."""
        if self.smooth_pos is None:
            self.smooth_pos = (float(pos[0]), float(pos[1]))
        else:
            a = SMOOTH_ALPHA
            self.smooth_pos = (
                a * pos[0] + (1.0 - a) * self.smooth_pos[0],
                a * pos[1] + (1.0 - a) * self.smooth_pos[1],
            )
        return int(round(self.smooth_pos[0])), int(round(self.smooth_pos[1]))

    def _move_cursor(self, target):
        """Só move o mouse se o deslocamento passar da zona morta. Retorna a posição atual."""
        last = self.last_cursor
        if last is None or abs(target[0] - last[0]) > DEADZONE_PX or abs(target[1] - last[1]) > DEADZONE_PX:
            pyautogui.moveTo(*target, duration=0)
            self.last_cursor = target
        return self.last_cursor

    def _confirm_gesture(self, raw_gesture, now):
        if raw_gesture != self.pending_gesture:
            self.pending_gesture = raw_gesture
            self.pending_since = now
        if now - self.pending_since >= self.gesture_confirm_seconds:
            return raw_gesture
        return self.current_gesture if self.current_gesture is not None else raw_gesture

    def _release_hold(self):
        if self.hold_gesture is not None:
            try:
                pyautogui.mouseUp()
            except Exception:
                print("[LunarVision] Erro ao soltar o botão do mouse:")
                traceback.print_exc()
            self.hold_gesture = None

    def _update_hold_gesture(self, gesture, cursor):
        if gesture == "arraste":
            if self.hold_gesture != gesture:
                self._release_hold()
                pyautogui.mouseDown(*cursor)
                self.hold_gesture = gesture
                self.status.set(f"Pressionando ({gesture}) em {cursor[0]}, {cursor[1]}")
        else:
            self._release_hold()

    def _update_click_gesture(self, gesture, cursor, now):
        # Um clique por "paz". Só rearma depois que a mão ficar fora do gesto por
        # `click_rearm_seconds`, então uma oscilação rápida não gera clique duplo.
        # `cursor` é o last_cursor (já filtrado pela zona morta), então o clique
        # cai onde o usuário mirou, mesmo se a ponta do dedo escorregar ao levantar o médio.
        if gesture == "clique":
            if self.click_armed and now - self.last_click_time >= self.click_cooldown_seconds:
                pyautogui.click(*cursor)
                self.last_click_time = now
                self.click_armed = False
                self.status.set(f"Clique: {cursor[0]}, {cursor[1]}")
            self.clique_released_at = None
        else:
            if self.clique_released_at is None:
                self.clique_released_at = now
            elif now - self.clique_released_at >= self.click_rearm_seconds:
                self.click_armed = True

    # --- Calibração ------------------------------------------------------------
    def start_calibration(self):
        if self.latest_shape is None:
            self.status.set("Conecte uma webcam antes de calibrar.")
            return
        self.cursor_enabled = False
        self._release_hold()
        self.points = []
        self.calibration_stage = 0
        self.calibrating = True
        self.status.set("Calibração iniciada. Clique no primeiro canto na imagem.")

    def finish_calibration(self):
        self.calibration.set_points(self.points[0], self.points[1])
        self.calibrating = False
        if not self.calibration.ready:
            self.status.set("Área inválida. Escolha pontos mais afastados.")
            messagebox.showwarning("Calibração", "Os dois pontos precisam formar uma área maior.")
            return
        self.cursor_enabled = True
        self._reset_tracking()
        self.status.set("Calibrado. Cursor ativo — veja os gestos disponíveis na tela.")

    def on_preview_click(self, event):
        if not self.calibrating or self.latest_shape is None:
            return
        h, w = self.latest_shape
        disp_w, disp_h = self.display_size
        if disp_w <= 0 or disp_h <= 0:
            return
        # A imagem fica centralizada no label e NÃO é esticada para preencher o
        # label: usa-se o tamanho real da imagem exibida (antes usava o do label,
        # o que deslocava um pouco os cantos da calibração).
        scale = disp_w / w
        offset_x = (self.preview.winfo_width() - disp_w) / 2
        offset_y = (self.preview.winfo_height() - disp_h) / 2
        x = int((event.x - offset_x) / scale)
        y = int((event.y - offset_y) / scale)
        if not (0 <= x < w and 0 <= y < h):
            return
        self.points.append((x, y))
        if len(self.points) == 1:
            self.calibration_stage = 1
            self.status.set("Primeiro ponto registrado. Clique no segundo canto.")
        elif len(self.points) == 2:
            self.finish_calibration()

    # --- Encerramento ----------------------------------------------------------
    def close(self):
        self.cursor_enabled = False
        self.stop_camera()
        self.hands.close()
        self.root.destroy()


if __name__ == "__main__":
    pyautogui.PAUSE = 0          # sem pausa artificial depois de cada ação do mouse
    pyautogui.FAILSAFE = False
    root = tk.Tk()
    app = LunarVisionApp(root)
    app.preview.bind("<Button-1>", app.on_preview_click)
    root.mainloop()