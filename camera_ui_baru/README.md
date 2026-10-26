# Camera UI Baru

Implementasi mandiri kamera dan antarmuka desktop yang terpisah dari `src/app.py`.

## Menjalankan

Dari root repository:

```powershell
python -m camera_ui_baru.app
python -m camera_ui_baru.app --source 1
python -m camera_ui_baru.app --source 0 --weights ""
```

Fitur:

- pembacaan kamera/video di thread background agar UI tidak macet;
- preview live;
- deteksi YOLO opsional dan bounding box;
- snapshot ke `runs/camera_ui_baru/snapshots`;
- rekaman MP4 ke `runs/camera_ui_baru/recordings`;
- kontrol source kamera tanpa mengubah implementasi lama.

