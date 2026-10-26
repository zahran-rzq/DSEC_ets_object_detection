<div align="center">

<a href="https://colab.research.google.com/github/zahran-rzq/DSEC_ets_object_detection/blob/arena/e1293095-dsec-ets-object-detection/DSEC_Security_Camera_ETS.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="52"></a>

# 🛡️ SecureVision — DSEC ETS · *Security Camera Application*

**Klik tombol ▶ Open in Colab di atas untuk langsung menjalankan notebook-nya.**

</div>

Aplikasi desktop kamera keamanan berbasis *deep learning* dengan model
**YOLO pre-trained** yang mendeteksi **person · car · truck · motorbike**
secara *real-time*, lalu **otomatis menyimpan gambar dan merekam video**
ketika kelas tersebut terdeteksi.

> Repositori ini berisi **laporan + notebook Colab + kode aplikasi** untuk
> menjawab **Soal 2 ETS (bobot 50%)** mata kuliah *Deep Learning for Security*.

---

## 🚀 Pilih jalur yang kamu butuhkan

| Tujuan | Yang dibuka | Waktu |
|---|---|---|
| **Mau langsung train & demo di Colab (waktu mepet)** | 👉 [`DSEC_Security_Camera_ETS.ipynb`](DSEC_Security_Camera_ETS.ipynb) — *Run all* | ±15 menit |
| **Mau baca / kumpulkan laporannya** | 👉 [`docs/LAPORAN_Soal2.md`](docs/LAPORAN_Soal2.md) · cetak PDF: [`docs/LAPORAN_Soal2.html`](docs/LAPORAN_Soal2.html) (Ctrl+P → Save as PDF) | — |
| **Mau jalankan aplikasi desktop di laptop** | `python -m src.app` | ±5 menit |
| **Mau lihat mockup antarmuka** | [`mockup/ui_mockup_light.png`](mockup/ui_mockup_light.png) | — |

### Colab — urutan tercepat saat ujian
1. Klik tombol **▶ Open in Colab** di paling atas halaman ini
   *(atau salin tautan berikut bila tombol tidak muncul):*

   ```
   https://colab.research.google.com/github/zahran-rzq/DSEC_ets_object_detection/blob/arena/e1293095-dsec-ets-object-detection/DSEC_Security_Camera_ETS.ipynb
   ```

   Cara kedua: buka <https://colab.research.google.com> → **File → Open notebook
   → tab GitHub** → ketik `zahran-rzq/DSEC_ets_object_detection` → pilih
   `DSEC_Security_Camera_ETS.ipynb`.
2. Setelah terbuka di Colab, pilih `Runtime → Change runtime type → **T4 GPU** → Save`.
3. `Runtime → Run all`.
4. **Bagian 5** = demo webcam *live* langsung dari browser (tanpa instal apa pun).
5. **Bagian 6** = unduh `best.pt` + arsip hasil.

> Tidak punya dataset? Tidak masalah. Model **pre-trained COCO** langsung
> mendeteksi keempat kelas target tanpa training. Bagian training memakai
> `coco8.yaml` hanya sebagai *smoke test* (~1 menit) agar pipeline terbukti jalan.

---

## 📁 Struktur proyek

```
DSEC_ets_object_detection/
├── DSEC_Security_Camera_ETS.ipynb   ⭐ notebook Colab: train + evaluasi + demo webcam
├── docs/
│   ├── LAPORAN_Soal2.md             laporan lengkap jawaban Soal 2
│   └── LAPORAN_Soal2.html           versi siap cetak → PDF (Ctrl+P)
├── mockup/
│   └── ui_mockup_light.png          mockup antarmuka, tema terang
├── src/
│   ├── app.py                       GUI desktop bertema terang (CustomTkinter)
│   ├── headless.py                  mode tanpa GUI (mini-PC / Raspberry Pi)
│   └── core/
│       ├── config.py                konfigurasi + 4 preset (balanced, max_recall, ...)
│       ├── detector.py              wrapper YOLO, filter kelas/conf/ROI, anotasi
│       ├── recorder.py              ring buffer pre-roll, penulis MP4/JPG/CSV
│       ├── pipeline.py              state machine pemicu rekaman
│       ├── camera.py                pembuka kamera tangguh (backend probe + watchdog)
│       └── metrics.py               IoU, Precision, Recall, AP, mAP (dari nol)
├── tests/test_pipeline.py           self-test logika tanpa model & tanpa kamera
├── tools/
│   ├── build_notebook.py            pembangkit notebook
│   ├── make_ui_mockup.py            pembangkit mockup UI
│   └── md_to_html.py                Markdown → HTML siap cetak
└── requirements.txt
```

---

## 💻 Menjalankan aplikasi desktop

```bash
git clone https://github.com/zahran-rzq/DSEC_ets_object_detection
cd DSEC_ets_object_detection
pip install -r requirements.txt          # Linux: sudo apt install python3-tk

python -m src.app                         # GUI tema terang, webcam 0
python -m src.app --source 1              # webcam kedua
python -m src.app --weights best.pt       # bobot hasil fine-tuning
python -m src.app --preset max_recall     # prioritas recall (anti penyusup lolos)
python -m src.app --backend DSHOW         # Windows: paksa backend DirectShow

python -m src.headless --source 0         # tanpa GUI
python -m src.headless --source video.mp4 # proses berkas video
python -m src.headless --source foto.jpg --save-frame

python -m tests.test_pipeline             # 19 pengujian logika, tanpa model
```

### 🔧 Troubleshooting: "Lampu kamera menyala, tapi tidak muncul di UI"

Gejala ini berarti kamera **terbuka** oleh OpenCV, tetapi tidak ada *frame* yang
masuk. Penyebabnya hampir selalu salah satu dari berikut:

| # | Penyebab | Solusi |
|---|---|---|
| 1 | Aplikasi lain memegang kamera (Zoom / Teams / Meet / browser / OBS) | **Tutup** aplikasi itu, lalu klik **Tes** |
| 2 | Backend OpenCV tidak cocok (umum di Windows: MSMF) | Klik **Tes** — aplikasi otomatis mencoba `DSHOW → MSMF → ANY`. Atau paksa: `python -m src.app --backend DSHOW` |
| 3 | Izin kamera untuk *desktop apps* nonaktif (Windows) | `Settings → Privacy & security → Camera` → nyalakan **Camera access**, **Let apps access your camera**, dan **Let desktop apps access your camera** |
| 4 | Resolusi 1280×720 tidak didukung | Turunkan: edit `cfg.camera.width/height` atau pakai preset `fast` (`--preset fast` → 640×480) |
| 5 | Salah indeks kamera (mis. kamera virtual OBS/DroidCam di indeks 0) | Klik **Tes** untuk melihat indeks yang benar, lalu pilih di dropdown **Sumber kamera** |
| 6 | Terpasang `opencv-python-headless` | `pip uninstall opencv-python-headless && pip install opencv-python` |
| 7 | OpenCV 5.x (Ultralytics resmi diuji di 4.x) | Kode sudah kompatibel; bila muncul error aneh saat inferensi: `pip install "opencv-python>=4.8,<5.0"` |

**Pemindaian kamera dari terminal:**

```bash
python -m src.core.camera          # atau: python -m src.headless --list-cameras
```

Aplikasi sekarang juga otomatis:
* mencoba beberapa backend dan hanya menerima kamera yang **benar-benar
  mengirim frame** (bukan sekadar `isOpened() == True`),
* menampilkan peringatan **"KAMERA TIDAK MENGIRIM FRAME — mencoba menyambung
  ulang..."** dan menyambung ulang otomatis,
* memberi tahu bila frame tampak **hitam** (lensa tertutup / dipakai aplikasi lain),
* memuat model & membuka kamera **di thread terpisah** sehingga GUI tidak beku.

---

Semua hasil tersimpan di `runs/capture/`:

```
runs/capture/
├── rec_20250115_203412_0001.mp4      # klip kejadian (pre-roll 3s + kejadian + post-roll 4s)
├── snapshots/snap_20250115_203412_person.jpg
└── events.csv                        # timestamp, event, classes, conf, n_objects, file
```

---

## 🔑 Inti jawaban (ringkas)

| Pertanyaan | Jawaban singkat |
|---|---|
| **Arsitektur model** | **YOLOv11-nano** (single-stage), pre-trained COCO. ~8–12 ms/frame → *real-time*; akurasi setara Faster R-CNN yang 15× lebih lambat. |
| **Konfigurasi 4 kelas** | Tiga lapis filter: ① `model.predict(..., classes=[0,2,3,7])` membuang 76 kelas lain **sebelum NMS**, ② ambang *confidence* per kelas (person paling sensitif 0.30), ③ filter luas kotak & ROI di level aplikasi. |
| **Alur sistem** | `cap.read()` → ring buffer (pre-roll) → letterbox 640 → forward pass → NMS → filter kelas/conf/ROI → *debounce* 3 frame → **kondisi pemicu** → simpan JPG + rekam MP4 → berhenti 4 s setelah objek hilang → catat ke `events.csv` (rotasi berkas 60 s, jeda antar event 5 s). |
| **Metrik terpenting** | ⭐ **RECALL**. Recall = TP/(TP+FN) adalah satu-satunya metrik yang berbanding terbalik langsung dengan jumlah penyusup lolos (*False Negative* — biaya fatal & permanen). **mAP** dipakai untuk *memilih model*, **Precision** dijaga lewat ROI/*debounce* agar petugas tidak *alarm fatigue*. Optimasi: threshold `person` rendah, IoU NMS 0.60, target **F2-score**. |
| **Antarmuka** | Tema terang: top bar (identitas + badge LIVE + jam) · feed video (kotak per kelas + HUD + titik REC berkedip + timer) · panel kanan (status, deteksi per kelas, penyimpanan, log) · baris kontrol (Start/Stop, Snapshot, Auto-Record, slider Confidence/IoU, preset, Folder Output). |

---

## 🧪 Hasil self-test

```
[1] State machine IDLE→DETECTING→RECORDING→COOLDOWN   ✅ 7/7
[2] Debounce (flicker 1 frame tidak merekam)          ✅
[3] Pemetaan kelas → COCO ids [0,2,3,7]               ✅ 3/3
[4] IoU (1.000 / 0.333 / 0.000)                       ✅ 3/3
[5] Precision / Recall / mAP (TP=2,FP=1,FN=1 → 0.667) ✅ 5/5
[6] Rotasi klip saat durasi maksimum                  ✅
                                    HASIL: 19 lulus, 0 gagal
```

---

*Dibuat untuk ETS mata kuliah Deep Learning for Security (DSEC).*
