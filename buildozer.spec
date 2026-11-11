[app]
title = Layered Player
package.name = layeredplayer
package.domain = org.nikol
source.dir = .
source.include_exts = py
version = 1.0

requirements = python3,kivy==2.3.0,numpy,soundfile

# Права: чтение музыки из памяти телефона
android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_AUDIO

android.api = 31
android.minapi = 24
android.archs = arm64-v8a
orientation = portrait
fullscreen = 0

# Не собирать библиотеки под архитектуры, которых нет
android.allow_backup = True

[buildozer]
log_level = 2
warn_on_root = 1
