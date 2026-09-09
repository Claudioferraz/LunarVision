import time
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


# --- Reconhecimento de gestos -------------------------------------------------
# Cada dedo é descrito pelo par (ponta, articulação PIP). Comparamos a distância
# de cada ponto até o pulso (landmark 0): se a ponta está bem mais longe do
# pulso do que a articulação do meio, consideramos o dedo "esticado".
FINGER_LANDMARKS = {
    "thumb": (4, 2),
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}
FINGER_EXTENSION_MARGIN = 1.1  # dedo esticado precisa estar ~10% mais longe do pulso que a articulação PIP
PINCH_THRESHOLD_RATIO = 0.35  # pinça: distância polegar-indicador menor que 35% do tamanho da mão


def _landmark_distance(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def finger_states(hand_landmarks):
    """Retorna {nome_do_dedo: True/False} indicando se cada dedo está esticado."""
    wrist = hand_landmarks[0]
    states = {}
    for name, (tip_idx, pip_idx) in FINGER_LANDMARKS.items():
        tip = hand_landmarks[tip_idx]
        pip = hand_landmarks[pip_idx]
        states[name] = _landmark_distance(tip, wrist) > _landmark_distance(pip, wrist) * FINGER_EXTENSION_MARGIN
    return states


def classify_gesture(hand_landmarks):
    """Classifica a mão em um dos modos:
        "desenho" -> polegar e indicador em pinça (mantém o botão pressionado).
        "clique"  -> indicador + médio esticados, os outros dobrados (sinal de "paz").
        "arraste" -> mão fechada, nenhum dedo esticado (mantém o botão pressionado).
        "pausa"   -> mão aberta, 4 ou mais dedos esticados (congela o cursor).
        "apontar" -> qualquer outra combinação (padrão: só move o cursor).
    """
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    hand_scale = _landmark_distance(wrist, middle_mcp) or 1e-6

    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    pinch_ratio = _landmark_distance(thumb_tip, index_tip) / hand_scale
    if pinch_ratio < PINCH_THRESHOLD_RATIO:
        return "desenho"

    states = finger_states(hand_landmarks)
    extended_count = sum(states.values())

    if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
        return "clique"
    if extended_count == 0:
        return "arraste"
    if extended_count >= 4:
        return "pausa"
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
        """Converte a posição da câmera para X/Y relativos entre 0% e 100%."""
        if not self.ready:
            return None
        percent_x = np.clip((x - self.x1) / (self.x2 - self.x1), 0.0, 1.0) * 100.0
        percent_y = np.clip((y - self.y1) / (self.y2 - self.y1), 0.0, 1.0) * 100.0
        return float(percent_x), float(percent_y)

    def percentages_to_screen(self, percent_x, percent_y, screen_w, screen_h):
        """Converte percentuais para a resolução atual, sem fixar 1080p ou 4K."""
        return (
            int(np.clip(percent_x / 100.0, 0.0, 1.0) * (screen_w - 1)),
            int(np.clip(percent_y / 100.0, 0.0, 1.0) * (screen_h - 1)),
        )

    def map_to_screen(self, x, y, screen_w, screen_h):
        percentages = self.to_percentages(x, y)
        if percentages is None:
            return None
        return self.percentages_to_screen(*percentages, screen_w, screen_h)


class LunarVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LunarVision — Protótipo")
        self.root.geometry("960x740")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.cap = None
        self.camera_options = []  # índice, backend, descrição
        self.camera_index = 0
        self.camera_backend = cv2.CAP_DSHOW
        self.running = False
        self.calibrating = False
        self.calibration_stage = 0
        self.points = []
        self.calibration = Calibration()
        self.latest_frame = None
        self.cursor_enabled = False

        # Estado do reconhecimento de gestos.
        self.current_gesture = None       # último gesto confirmado
        self.pending_gesture = None       # gesto bruto detectado neste instante (antes de confirmar)
        self.pending_since = 0.0
        self.gesture_confirm_seconds = 0.08  # tempo mínimo pra confirmar um gesto (filtra ruído/flicker)
        self.hold_gesture = None          # 'arraste' ou 'desenho' enquanto o botão está pressionado
        self.last_click_time = 0.0
        self.click_cooldown_seconds = 0.5  # tempo mínimo entre dois cliques do gesto "clique"

        self.video_timestamp_ms = 0
        self.hands = self.create_hand_tracker()

        self.build_ui()
        self.scan_cameras()

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
                "da área da lousa na imagem. O cursor sempre segue o dedo indicador. Gestos: só o indicador = "
                "apontar; indicador+médio ('paz') = clique; mão fechada = arraste; pinça (polegar+indicador) = "
                "desenho; mão aberta = pausa (congela o cursor).")
        ttk.Label(self.root, text=info, wraplength=920, padding=10).pack(anchor="w")
        self.preview = ttk.Label(self.root)
        self.preview.pack(expand=True, fill="both", padx=10, pady=10)

    def scan_cameras(self):
        self.stop_camera()
        self.status.set("Procurando webcams com DirectShow e Media Foundation...")
        found = []
        for index in range(10):
            for backend, backend_name in ((cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation")):
                test_cap = cv2.VideoCapture(index, backend)
                if test_cap.isOpened():
                    ok, frame = test_cap.read()
                    if ok and frame is not None and frame.size:
                        found.append((index, backend, f"Índice {index} — {backend_name}"))
                test_cap.release()
        self.camera_options = found
        self.camera_combo["values"] = [item[2] for item in found]
        if found:
            self.camera_combo.current(0)
            self.status.set(f"{len(found)} opção(ões) encontrada(s). Pressione Conectar.")
        else:
            self.camera_combo.set("")
            self.status.set("Nenhuma webcam encontrada. Verifique o driver e feche outros aplicativos que usam a câmera.")

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
        self.running = True
        self.status.set("Webcam conectada. Pressione Calibrar.")
        self.update_frame()

    def update_frame(self):
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.status.set("A webcam conectou, mas não entregou imagem.")
            self.root.after(50, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        self.latest_frame = frame.copy()
        display = frame.copy()

        if self.calibrating:
            text = "Clique no CANTO SUPERIOR ESQUERDO" if self.calibration_stage == 0 else "Clique no CANTO INFERIOR DIREITO"
            cv2.putText(display, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        for index, point in enumerate(self.points):
            cv2.circle(display, point, 10, (0, 255, 0), -1)
            cv2.putText(display, str(index + 1), (point[0] + 12, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.calibration.ready:
            cv2.rectangle(display, (self.calibration.x1, self.calibration.y1), (self.calibration.x2, self.calibration.y2), (255, 0, 0), 2)

        detection = self.detect_hand(frame, display)
        if detection is not None:
            (tx, ty), raw_gesture = detection
            cv2.circle(display, (tx, ty), 12, (0, 0, 255), 2)
            cv2.circle(display, (tx, ty), 4, (255, 255, 255), -1)
            percentages = self.calibration.to_percentages(tx, ty)
            mapped = None
            if percentages is not None:
                screen_w, screen_h = pyautogui.size()
                mapped = self.calibration.percentages_to_screen(*percentages, screen_w, screen_h)
            now = time.monotonic()
            gesture = self._confirm_gesture(raw_gesture, now)
            if mapped and percentages is not None and self.cursor_enabled:
                try:
                    if gesture != "pausa":
                        pyautogui.moveTo(*mapped, duration=0)
                    self._update_hold_gesture(gesture, mapped)
                    self._update_click_gesture(gesture, mapped, now)
                except Exception:
                    print("[LunarVision] Erro ao mover/clicar com pyautogui:")
                    traceback.print_exc()
                    self.status.set("Erro no pyautogui — veja o console.")
                cv2.putText(display, f"Lousa: X {percentages[0]:.1f}% | Y {percentages[1]:.1f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                cv2.putText(display, f"Tela atual: {mapped[0]}, {mapped[1]}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                cv2.putText(display, f"Gesto: {gesture}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
            self.current_gesture = gesture
        else:
            self._release_hold()
            self.current_gesture = None
            self.pending_gesture = None

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((920, 600))
        photo = ImageTk.PhotoImage(image=image)
        self.preview.configure(image=photo)
        self.preview.image = photo
        self.root.after(15, self.update_frame)

    def create_hand_tracker(self):
        model_path = Path(__file__).with_name("hand_landmarker.task")
        if not model_path.exists():
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            if hasattr(self, "status"):
                self.status.set("Baixando o modelo de detecção da mão...")
            try:
                urllib.request.urlretrieve(url, model_path)
            except Exception as exc:
                raise RuntimeError(f"Não foi possível baixar hand_landmarker.task: {exc}") from exc

        # Verifica se o modelo baixado não está corrompido/incompleto (download parcial
        # é uma causa comum de "nenhuma mão detectada" sem erro nenhum aparecendo).
        min_expected_bytes = 1_000_000
        if model_path.exists() and model_path.stat().st_size < min_expected_bytes:
            print(f"[LunarVision] hand_landmarker.task parece corrompido "
                  f"({model_path.stat().st_size} bytes). Apagando para baixar de novo.")
            model_path.unlink()
            return self.create_hand_tracker()

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.40,
            min_hand_presence_confidence=0.40,
            min_tracking_confidence=0.40,
        )
        print(f"[LunarVision] Modelo carregado de {model_path} "
              f"({model_path.stat().st_size} bytes).")
        return vision.HandLandmarker.create_from_options(options)

    def _confirm_gesture(self, raw_gesture, now):
        """Exige que o gesto bruto se mantenha por `gesture_confirm_seconds` antes de
        confirmá-lo, para evitar que um único frame ruidoso dispare uma ação errada."""
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

    def _update_hold_gesture(self, gesture, mapped):
        """Gestos 'arraste' e 'desenho' mantêm o botão esquerdo pressionado enquanto
        o gesto continuar; qualquer outro gesto solta o botão."""
        if gesture in ("arraste", "desenho"):
            if self.hold_gesture != gesture:
                self._release_hold()  # solta qualquer pressão anterior, por segurança
                pyautogui.mouseDown(*mapped)
                self.hold_gesture = gesture
                self.status.set(f"Pressionando ({gesture}) em {mapped[0]}, {mapped[1]}")
        else:
            self._release_hold()

    def _update_click_gesture(self, gesture, mapped, now):
        """Gesto 'clique' dispara um clique único na transição (borda de subida),
        respeitando um pequeno cooldown para não disparar cliques repetidos."""
        if (gesture == "clique" and self.current_gesture != "clique"
                and now - self.last_click_time >= self.click_cooldown_seconds):
            pyautogui.click(*mapped)
            self.last_click_time = now
            self.status.set(f"Clique: {mapped[0]}, {mapped[1]}")

    def detect_hand(self, frame, display):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.video_timestamp_ms += 33
            result = self.hands.detect_for_video(mp_image, self.video_timestamp_ms)
        except Exception:
            # Antes essa exceção matava o loop do update_frame em silêncio.
            # Agora ela aparece no console para sabermos exatamente o que houve.
            print("[LunarVision] Erro na detecção da mão:")
            traceback.print_exc()
            self.status.set("Erro na detecção da mão — veja o console.")
            return None

        if not result.hand_landmarks:
            self._log_no_hand_once()
            return None

        hand = result.hand_landmarks[0]
        h, w = frame.shape[:2]
        # Desenha a malha mínima para confirmar visualmente que a mão foi encontrada.
        for landmark in hand:
            px, py = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(display, (px, py), 2, (0, 255, 0), -1)
        tip = hand[8]  # índice 8: ponta do dedo indicador
        gesture = classify_gesture(hand)
        return (int(tip.x * w), int(tip.y * h)), gesture

    def _log_no_hand_once(self):
        # Log leve (1x/seg) para confirmar que o loop está rodando e chamando
        # a detecção, mas que a MediaPipe simplesmente não está achando a mão.
        now = time.monotonic()
        if now - getattr(self, "_last_no_hand_log", 0) > 1.0:
            self._last_no_hand_log = now
            print("[LunarVision] Nenhuma mão detectada neste frame.")

    def start_calibration(self):
        if self.latest_frame is None:
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
        self.current_gesture = None
        self.pending_gesture = None
        self.status.set("Calibrado. Cursor ativo — veja os gestos disponíveis na tela.")

    def on_preview_click(self, event):
        if not self.calibrating or self.latest_frame is None:
            return
        h, w = self.latest_frame.shape[:2]
        displayed_w = max(1, self.preview.winfo_width())
        displayed_h = max(1, self.preview.winfo_height())
        scale = min(displayed_w / w, displayed_h / h)
        offset_x = (displayed_w - w * scale) / 2
        offset_y = (displayed_h - h * scale) / 2
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

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def close(self):
        self._release_hold()
        self.stop_camera()
        self.hands.close()
        self.root.destroy()


if __name__ == "__main__":
    pyautogui.PAUSE = 0.01
    pyautogui.FAILSAFE = False
    root = tk.Tk()
    app = LunarVisionApp(root)
    app.preview.bind("<Button-1>", app.on_preview_click)
    root.mainloop()
