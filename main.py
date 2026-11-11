# -*- coding: utf-8 -*-
"""
Многослойный плеер (Android/Kivy + ПК-тест).

Загружаешь синхронные версии трека (слои) и плавно переключаешь их
кнопками на экране (клавиши 1..9 на ПК). Кроссфейд 0.8 с, есть loop.

Аудио-движок сам выбирает бэкенд:
  * ПК (Windows/Linux/macOS)  -> sounddevice (PortAudio)
  * Android                   -> AudioTrack через pyjnius
"""

import os
import threading
import numpy as np

FADE_SEC = 0.8
BLOCK = 2048  # размер аудиоблока на Android, кадров


# ----------------------------------------------------------------------
# Загрузка аудио (soundfile, а на Android — запасной вариант через wave)
# ----------------------------------------------------------------------

def read_audio(path):
    """Возвращает (data (frames, 2) float32, samplerate)."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except ImportError:
        # Запасной путь: стандартный wave (только WAV)
        import wave
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width == 2:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError("Поддерживаются WAV 16/32 бит")
        data = data.reshape(-1, ch)
        if ch == 1:
            data = np.repeat(data, 2, axis=1)
        elif ch > 2:
            data = data[:, :2]
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    return np.ascontiguousarray(data, dtype=np.float32), sr


# ----------------------------------------------------------------------
# Ядро микширования (одно и то же для ПК и Android)
# ----------------------------------------------------------------------

class Layer:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.volume = 1.0
        self.gain = 0.0
        self.target = 0.0
        self.fading = False

    @property
    def is_on(self):
        return self.target > 0.0


class AudioEngine:
    def __init__(self):
        self.sr = None
        self.layers = []
        self.total = 0
        self.pos = 0
        self.playing = False
        self.finished = False
        self.loop = True

    # --- загрузка ---

    def add_files(self, paths):
        added, skipped = [], []
        for path in paths:
            try:
                data, sr = read_audio(path)
            except Exception as e:
                skipped.append("%s — %s" % (os.path.basename(path), e))
                continue
            if self.sr is None:
                self.sr = sr
            elif sr != self.sr:
                # линейная ресемплизация в частоту первого файла
                ratio = self.sr / sr
                n = int(len(data) * ratio)
                idx = np.linspace(0, len(data) - 1, n)
                c = np.empty((n, 2), dtype=np.float32)
                for chn in range(2):
                    c[:, chn] = np.interp(idx, np.arange(len(data)), data[:, chn])
                data = c
                skipped.append("%s — пересемплирован" % os.path.basename(path))
            self.layers.append(Layer(os.path.basename(path), data))
            added.append(path)
        if self.layers:
            self.total = max(len(l.data) for l in self.layers)
            for l in self.layers:
                if len(l.data) < self.total:
                    l.data = np.vstack(
                        [l.data, np.zeros((self.total - len(l.data), 2), np.float32)]
                    )
        return added, skipped

    def clear(self):
        self.stop()
        self.layers = []
        self.sr = None
        self.total = 0
        self.pos = 0

    # --- управление ---

    def toggle(self, index):
        if 0 <= index < len(self.layers):
            layer = self.layers[index]
            layer.target = 0.0 if layer.is_on else layer.volume
            layer.fading = True

    def set_volume(self, index, volume):
        layer = self.layers[index]
        layer.volume = volume
        if layer.is_on:
            layer.target = volume
            layer.fading = True

    def play(self):
        if not self.layers:
            return
        if self.pos >= self.total:
            self.pos = 0
        self.playing = True
        self.finished = False

    def pause(self):
        self.playing = False

    def toggle_play(self):
        self.pause() if self.playing else self.play()

    def stop(self):
        self.playing = False
        self.pos = 0
        self.finished = False

    def seek(self, frame):
        self.pos = max(0, min(frame, self.total))

    # --- рендер одного блока звука (используют оба бэкенда) ---

    def render(self, frames):
        out = np.zeros((frames, 2), dtype=np.float32)
        if not (self.playing and self.layers and self.total > 0):
            return out

        fade_frames = max(1, int(FADE_SEC * self.sr))
        step = 1.0 / fade_frames
        written = 0
        start = self.pos

        while written < frames:
            avail = min(frames - written, self.total - start)
            if avail <= 0:
                if self.loop:
                    start = 0
                    continue
                self.playing = False
                self.finished = True
                self.pos = self.total
                break

            seg = np.zeros((avail, 2), dtype=np.float32)
            for layer in self.layers:
                g0 = layer.gain
                g1 = g0
                if layer.fading:
                    direction = 1.0 if layer.target > g0 else -1.0
                    g1 = g0 + direction * step * avail
                    if (direction > 0 and g1 >= layer.target) or (
                        direction < 0 and g1 <= layer.target
                    ):
                        g1 = layer.target
                        layer.fading = False
                    g1 = max(0.0, min(g1, 1.0))
                if g0 > 0.0 or g1 > 0.0:
                    ramp = np.linspace(g0, g1, avail, endpoint=False)
                    seg += layer.data[start:start + avail] * ramp[:, None]
                layer.gain = g1

            np.clip(seg, -1.0, 1.0, out=seg)
            out[written:written + avail] = seg
            written += avail
            start += avail

        if not self.finished:
            self.pos = start if start < self.total else 0
        return out


# ----------------------------------------------------------------------
# Бэкенды вывода звука
# ----------------------------------------------------------------------

class DesktopBackend:
    """ПК: PortAudio через sounddevice."""

    def __init__(self):
        self.stream = None

    def start(self, engine):
        import sounddevice as sd

        def callback(outdata, frames, time_info, status):
            outdata[:] = engine.render(frames)

        self.stream = sd.OutputStream(
            samplerate=engine.sr, channels=2, dtype="float32",
            blocksize=0, callback=callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class AndroidBackend:
    """Android: AudioTrack (PCM 16 бит) в отдельном потоке."""

    def __init__(self):
        self.track = None
        self.running = False
        self.thread = None

    def start(self, engine):
        from jnius import autoclass
        AudioTrack = autoclass("android.media.AudioTrack")
        AudioManager = autoclass("android.media.AudioManager")
        AudioFormat = autoclass("android.media.AudioFormat")

        buf = AudioTrack.getMinBufferSize(
            engine.sr, AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_16BIT
        )
        self.track = AudioTrack(
            AudioManager.STREAM_MUSIC, engine.sr,
            AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_16BIT,
            max(buf, BLOCK * 4), AudioTrack.MODE_STREAM,
        )
        self.track.play()

        self.running = True

        def worker():
            while self.running:
                block = engine.render(BLOCK)
                pcm = (block * 32767.0).astype("<i2").tobytes()
                self.track.write(pcm, 0, len(pcm))

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
        if self.track:
            self.track.stop()
            self.track.release()
            self.track = None


def make_backend():
    try:
        from jnius import autoclass  # noqa: F401 — есть только на Android
        return AndroidBackend()
    except Exception:
        return DesktopBackend()


# ----------------------------------------------------------------------
# Интерфейс Kivy
# ----------------------------------------------------------------------

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.slider import Slider
from kivy.utils import platform

KV = """
<Root>:
    orientation: 'vertical'
    padding: dp(8)
    spacing: dp(6)

    BoxLayout:
        size_hint_y: None
        height: dp(50)
        spacing: dp(6)
        Button:
            text: '+ Загрузить файлы'
            on_release: root.open_chooser()
        Button:
            text: 'Очистить'
            on_release: root.clear_all()

    ScrollView:
        BoxLayout:
            id: layers_box
            orientation: 'vertical'
            size_hint_y: None
            height: self.minimum_height
            padding: dp(4)
            spacing: dp(6)

    Label:
        id: time_label
        size_hint_y: None
        height: dp(24)
        text: '0:00 / 0:00'

    Slider:
        id: seek
        size_hint_y: None
        height: dp(30)
        range: 0, 0

    BoxLayout:
        size_hint_y: None
        height: dp(56)
        spacing: dp(6)
        CheckBox:
            id: loop_box
            active: True
            size_hint_x: None
            width: dp(40)
            on_active: root.engine_loop(self.active)
        Label:
            text: 'Повтор'
            size_hint_x: None
            width: dp(70)
        Button:
            id: play_btn
            text: 'Играть'
            on_release: root.toggle_play()
        Button:
            text: 'Стоп'
            on_release: root.stop_all()
"""


class Root(BoxLayout):
    pass


class PlayerApp(App):
    def build(self):
        self.title = "Слоёный плеер"
        self.engine = AudioEngine()
        self.backend = make_backend()
        self.backend_started = False

        self.root = Root()
        self.root.engine_loop = lambda active: setattr(self.engine, "loop", active)

        self.time_label = self.root.ids.time_label
        self.seek = self.root.ids.seek
        self.play_btn = self.root.ids.play_btn
        self.layers_box = self.root.ids.layers_box
        self.layer_widgets = []  # (button, slider)

        self.seek.bind(on_touch_down=self._seek_start)
        self.seek.bind(value=self._seek_move)

        Clock.schedule_interval(self.update_ui, 0.1)

        if platform not in ("android", "ios"):
            Window.bind(on_key_down=self._on_key_down)

        return self.root

    # --- загрузка файлов ---

    def open_chooser(self):
        if platform == "android":
            self._request_permissions()
            start_path = "/storage/emulated/0"
        else:
            start_path = os.path.expanduser("~")

        chooser = FileChooserListView(path=start_path, filters=["*.wav", "*.flac", "*.ogg", "*.mp3"], multiselect=True)
        box = BoxLayout(orientation="vertical", spacing=dp(6) if False else 6)
        box.add_widget(chooser)

        btns = BoxLayout(size_hint_y=None, height=50, spacing=6)
        ok = Button(text="Выбрать")
        cancel = Button(text="Отмена")
        btns.add_widget(ok)
        btns.add_widget(cancel)
        box.add_widget(btns)

        popup = Popup(title="Выбери слои", content=box, size_hint=(0.95, 0.9))
        cancel.bind(on_release=popup.dismiss)

        def do_load(*a):
            popup.dismiss()
            if chooser.selection:
                self.load_files(chooser.selection)

        ok.bind(on_release=do_load)
        popup.open()

    def _request_permissions(self):
        try:
            from android.permissions import request_permissions, Permission
            request_permissions([
                Permission.READ_EXTERNAL_STORAGE,
                Permission.READ_MEDIA_AUDIO,
            ])
        except Exception:
            pass

    def load_files(self, paths):
        self.engine.add_files(paths)
        self.rebuild_layers()
        self.seek.range = (0, self.engine.total)

    def rebuild_layers(self):
        self.layers_box.clear_widgets()
        self.layer_widgets = []
        for i, layer in enumerate(self.engine.layers):
            key = str(i + 1) if i < 9 else "–"

            btn = Button(
                text="[%s]  %s" % (key, layer.name),
                size_hint_y=None,
                height=48,
            )
            btn.bind(on_release=lambda _b, idx=i: self.engine.toggle(idx))

            row = BoxLayout(size_hint_y=None, height=48, spacing=6)
            row.add_widget(btn)
            slider = Slider(min=0, max=100, value=100, width=140)
            slider.bind(
                value=lambda _s, v, idx=i: self.engine.set_volume(idx, v / 100.0)
            )
            row.add_widget(slider)
            self.layers_box.add_widget(row)
            self.layer_widgets.append((btn, slider))

    def clear_all(self):
        self.engine.clear()
        self.rebuild_layers()
        self.seek.range = (0, 0)
        self.time_label.text = "0:00 / 0:00"

    # --- воспроизведение ---

    def toggle_play(self):
        if not self.backend_started and self.engine.sr:
            self.backend.start(self.engine)
            self.backend_started = True
        self.engine.toggle_play()

    def stop_all(self):
        self.engine.stop()

    # --- перемотка ---

    def _seek_start(self, *_a):
        self.was_playing = self.engine.playing

    def _seek_move(self, *_a):
        self.engine.seek(int(self.seek.value))

    # --- обновление интерфейса ---

    def _fmt(self, frames):
        if not self.engine.sr:
            return "0:00"
        sec = int(frames // self.engine.sr)
        return "%d:%02d" % divmod(sec, 60)

    def update_ui(self, *_a):
        self.seek.value = self.engine.pos
        self.time_label.text = "%s / %s" % (
            self._fmt(self.engine.pos), self._fmt(self.engine.total),
        )
        self.play_btn.text = "Пауза" if self.engine.playing else "Играть"

        for i, (btn, _slider) in enumerate(self.layer_widgets):
            if i < len(self.engine.layers):
                layer = self.engine.layers[i]
                name = layer.name
                key = str(i + 1) if i < 9 else "–"
                mark = "ON " if layer.is_on else "off"
                btn.text = "[%s] %s  (%s)" % (key, name, mark)
                btn.background_color = (
                    (0.3, 0.8, 0.4, 1) if layer.is_on else (0.8, 0.3, 0.3, 1)
                )

    # --- горячие клавиши (только ПК) ---

    def _on_key_down(self, _window, key, *_a):
        try:
            ch = chr(key)
        except Exception:
            return
        if ch.isdigit() and 1 <= int(ch) <= 9:
            self.engine.toggle(int(ch) - 1)
        elif key == 32:      # Space
            self.toggle_play()
        elif ch in ("s", "S", "ы", "Ы"):
            self.stop_all()


if __name__ == "__main__":
    PlayerApp().run()
