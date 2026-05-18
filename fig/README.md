# Figures

Paper figures for the SEABAD dataset paper.

| File | Figure | Description |
|------|--------|-------------|
| `mybad_curation.pdf` / `mybad_curation.png` | Figure 1 | Dual-branch curation pipeline. **Positive-label branch (left):** six sequential stages — metadata acquisition, download to FLAC, acoustic deduplication, segment extraction, species balancing, quality assurance — yielding 25,000 bird-present clips. **Negative-label branch (right):** six parallel source-specific extractions — BirdVox-DCASE-20k (9,983), Freefield1010 (5,755), Warblrb10k (1,950), FSC-22 (1,875), ESC-50 (1,840), DataSEC (3,597) — yielding 25,000 bird-absent clips. Both branches produce 3-second, 16 kHz mono WAV files. |
| `species_balance.png` | Figure 2 | Species-clip distribution before and after diversity-aware balancing. Pre-balancing: 38,481 clips, 1,677 species, mean 22.9 clips/species, Gini 0.601. Post-balancing: 25,000 clips, 1,677 species, mean 14.9 clips/species, Gini 0.519 (13.7% reduction). |
| `fig_geographic_distribution.pdf` | Figure 3 | Hexbin map of 18,999 positive recordings within the SE Asia bounding box (latitudes −9.97° to 19.98°, longitudes 95.37° to 124.88°). Country breakdown: Malaysia n=8,196 (43.1%), Thailand n=5,207 (27.4%), Indonesia n=4,188 (22.0%), Singapore n=1,376 (7.2%), Brunei n=32 (0.2%). Of the remaining 6,001 clips, 602 (2.4%) lacked coordinates and 5,399 (21.6%) fell outside the bounding box. |
| `qa_spectrograms_sample.png` | Figure 4 | 5×5 grid of mel spectrograms (4K, 3840×2160) generated for manual QA auditing. Shows acoustic diversity across positive samples: tonal calls, trills, repeated note sequences, and complex song structures. Computed with 80 mel bins, 512-point FFT, hop length 128, 0–8 kHz, 3-second clips at 16 kHz. |
