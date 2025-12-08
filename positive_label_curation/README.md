# XC → 3-Second Clips Extraction Pipeline  
**Goal**: From thousands of Xeno-canto FLAC recordings → extract high-quality, non-silent 3-second clips  
**Final dataset (when using `--flatten`)**:  
- Exactly **25,000 loudest clips** in the main folder  
- **≤99 clips** in `quarantine/` (the "almost made it" ones)  
- Everything else permanently deleted

---

### 1. Input Structure

FLAC files from cleaned MP3

```text
INPUT_ROOT/
├── Species_A/
│   ├── xc12345.flac
│   └── xc67890.flac
├── Species_B/
└── ...
```

### 2. Per-File Clip Selection Strategy

| Recording Duration | Expected Clips | Rule |
|--------------------|----------------|------|
| < 3.0 s            | 0              | Skip |
| 3.0 – 6.0 s        | 1              | Best 3s segment |
| 6.0 – 12.0 s       | 2              | Two diverse segments |
| ≥ 12.0 s           | 2              | Skip first 3s (avoid announcer), then pick 2 diverse |

→ Uses **sliding 3.0s windows** with **100 ms step**

### 3. Clip Selection Algorithm (per file)

1. Compute RMS for every 3s window
2. Keep only windows with **RMS ≥ threshold** (default: 0.003)
3. Sort candidates by RMS descending
4. Greedily pick highest → remove all windows within ±1.5 s (temporal diversity)
5. Repeat until target reached
6. If `--guarantee` → fall back to best windows even below threshold

### 4. Audio Processing per Clip

- Extract exact 3.0s segment
- Detect clipping (`|sample| ≥ 0.9999`)
- If clipped → peak scale to 0.99 + soft tanh limiter (α = 5.0)
- Save as **16 kHz, 16-bit PCM WAV**

### 5. Output Modes

| Mode       | Placement                                    | Example Filename           |
|------------|----------------------------------------------|----------------------------|
| Default    | `output_root/Species_Name/...`               | Hierarchical               |
| `--flatten`| `output_root/xc12345_1500.wav`               | Flat (recommended)         |

### 6. Global Post-Processing (`--flatten` only)

**Single source of truth**: `clips_log.csv`

→ Load CSV (ignore anything already in quarantine/)
→ Sort all clips globally by RMS (descending)
→ Top 25,000     → stay/move to main folder
→ Next 99        → move to quarantine/
→ All others     → permanently deleted
→ Overwrite CSV with final 25,099 rows

Final structure:

```text
output_root/
├── xc00001_2300.wav
├── ... (exactly 25,000 files)
└── quarantine/           ← ≤99 files
    └── xc98765_1200.wav



