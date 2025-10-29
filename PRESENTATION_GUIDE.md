# TopoFlow Presentation Guide

**Created for:** PhD Defense / Research Presentation
**Author:** Ammar Kheddar
**Date:** October 2025
**Project:** Physics-Informed Deep Learning for Multi-Pollutant Air Quality Forecasting

---

## 📂 Presentation Files

Two HTML presentation slides have been created for your research presentation:

### 1. Methodology Slide
**File:** `presentation_methodology.html`

**Content:**
- **Wind-Guided Scanning Innovation** - Shows how patches are reordered based on wind direction
- **Elevation-Based Attention** - Explains terrain-aware attention bias mechanism
- **Multi-Pollutant Modeling** - Demonstrates simultaneous prediction of 6 pollutant species
- **Large-Scale Training** - Highlights distributed training on 800 GPUs
- **Architecture Flow** - Animated diagram showing the complete model pipeline

**Key Features:**
- Beautiful gradient animations
- Color-coded innovation boxes
- Mathematical formulas for physics-based biases
- Step-by-step architecture visualization
- Professional color scheme (purple/blue gradient)

---

### 2. Results Slide
**File:** `presentation_results.html`

**Content:**
- **China Region Map** - Animated visualization of the study area (128×256 grid)
- **Performance Metrics** - Detailed RMSE/MAE for all 6 pollutants across 4 horizons
- **Training Statistics** - Infrastructure details (800 GPUs, 48h training, etc.)
- **Validation Loss** - Best checkpoint: 0.3557
- **Coverage Information** - 34.5% China coverage (11,317 active pixels)

**Key Features:**
- Animated China map with grid overlay
- Color-coded pollutant cards (red for PM2.5, orange for PM10, etc.)
- Pulsing region animation
- Performance summary dashboard
- Professional color scheme (green/teal gradient)

---

## 🚀 How to Use the Presentations

### Opening the Files

**Option 1: Local Browser** (Recommended)
1. Copy both HTML files to your local machine
2. Double-click to open in any web browser (Chrome, Firefox, Safari, Edge)
3. Presentations will auto-animate on load

**Option 2: From LUMI**
```bash
# View the files on LUMI
cd /scratch/project_462000640/ammar/aq_net2
firefox presentation_methodology.html &  # If X11 forwarding is enabled
firefox presentation_results.html &
```

**Option 3: Export to PDF**
1. Open in browser
2. Press Ctrl+P (or Cmd+P on Mac)
3. Select "Save as PDF"
4. Adjust settings for best quality

---

## 📊 Presentation Flow Suggestion

### Slide 1: Methodology (3-5 minutes)

**Opening:**
> "TopoFlow is a physics-informed deep learning architecture for multi-pollutant air quality forecasting. It introduces several key innovations..."

**Points to Cover:**
1. **Wind-Guided Scanning** (~1 min)
   - Patches reordered based on wind direction (upwind → downwind)
   - 16 pre-computed wind sectors
   - Captures atmospheric transport patterns

2. **Elevation-Based Attention** (~1.5 min)
   - Terrain-aware bias penalizes uphill pollutant transport
   - Formula: bias = -α × ReLU((elev_j - elev_i) / H_scale)
   - Applied **before** softmax (additive, not multiplicative)
   - Learnable parameter α (initialized to 0.0)

3. **Multi-Pollutant Modeling** (~1 min)
   - Simultaneous prediction of 6 species: PM2.5, PM10, SO₂, NO₂, CO, O₃
   - Multi-horizon: 12h, 24h, 48h, 96h
   - China region mask (128×256 grid)

4. **Large-Scale Training** (~1 min)
   - 800 AMD MI250X GPUs on LUMI supercomputer
   - 100 nodes × 8 GPUs/node
   - PyTorch Lightning DDP
   - ~48 hours training time

**Transition:**
> "With this physics-informed architecture, we achieved the following results..."

---

### Slide 2: Results (3-5 minutes)

**Opening:**
> "We evaluated TopoFlow on the test year 2018 across the China and Taiwan region..."

**Points to Cover:**

1. **Study Region** (~1 min)
   - 128×256 spatial grid
   - 11,317 active pixels (34.5% coverage)
   - China + Taiwan region
   - Point to animated map

2. **Performance Metrics** (~2 min)
   - Best validation loss: **0.3557**
   - Test set: 4,000 samples from 2018

   **Highlight Best Results:**
   - **PM2.5**: RMSE from 10.72 µg/m³ (12h) to 13.40 µg/m³ (96h)
   - **SO₂**: Excellent accuracy, RMSE ~2.8-3.3 µg/m³ across all horizons
   - **NO₂**: Consistent performance, RMSE ~9.2-10.1 µg/m³

   **Interpretation:**
   - Short-term forecasts (12h-24h) are most accurate
   - Longer horizons (96h) show expected degradation
   - All pollutants show stable multi-horizon prediction

3. **Training Infrastructure** (~1 min)
   - 100 nodes on LUMI
   - 800 GPUs (AMD MI250X)
   - 48 hours training
   - 20,000 optimization steps

**Closing:**
> "TopoFlow demonstrates that physics-informed inductive biases—wind-guided scanning and elevation-aware attention—enable accurate multi-pollutant air quality forecasting at scale."

---

## 🎨 Technical Details

### Methodology Slide

**Innovations Explained:**

1. **Wind-Guided Scanning**
   - **Problem:** Standard transformers process patches in arbitrary order
   - **Solution:** Reorder patches along wind direction (upwind → downwind)
   - **Implementation:** Pre-compute 16 sector orderings, select dynamically per batch
   - **Impact:** Captures atmospheric transport physics

2. **Elevation-Based Attention**
   - **Problem:** Pollutants struggle to climb mountains (physics!)
   - **Solution:** Add elevation bias to attention scores **before softmax**
   - **Formula:** bias_ij = -α × ReLU((elev_j - elev_i) / H_scale)
   - **Why additive?** Maintains attention normalization (sum = 1)
   - **Learnable:** α starts at 0.0, learns optimal penalty strength

3. **Architecture Flow**
   ```
   Input Data → Wind Scan → Block 0 (Physics) → Blocks 1-5 (Standard) → Decoder
   ```

---

### Results Slide

**Performance Breakdown:**

| Pollutant | 12h RMSE | 24h RMSE | 48h RMSE | 96h RMSE | Unit |
|-----------|----------|----------|----------|----------|------|
| **PM2.5** | 10.72 | 12.67 | 12.21 | 13.40 | µg/m³ |
| **PM10** | 17.90 | 20.29 | 19.85 | 21.63 | µg/m³ |
| **SO₂** | 2.89 | 2.88 | 2.81 | 3.27 | µg/m³ |
| **NO₂** | 9.75 | 9.16 | 9.33 | 10.13 | µg/m³ |
| **CO** | 48.98 | 48.06 | 48.65 | 54.02 | µg/m³ |
| **O₃** | 24.06 | 19.97 | 21.13 | 21.44 | µg/m³ |

**Key Observations:**
- SO₂ shows most stable performance across horizons
- PM2.5/PM10 show expected correlation (coarse vs. fine particles)
- O₃ peaks at 12h (photochemical complexity)
- All metrics in µg/m³ (micrograms per cubic meter)

---

## 🔧 Customization

### Changing Colors

**Methodology Slide:**
```css
/* Line 8-9: Main gradient */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Change to blue/green: */
background: linear-gradient(135deg, #667eea 0%, #11998e 100%);
```

**Results Slide:**
```css
/* Line 8-9: Main gradient */
background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);

/* Change to purple: */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### Adding Your Logo

Add this inside `<div class="slide">`:
```html
<img src="your_logo.png" style="position: absolute; top: 20px; right: 20px; height: 60px;">
```

### Adjusting Animation Speed

Find `animation-delay` values and modify:
```css
.innovation-box:nth-child(1) { animation-delay: 0.2s; }  /* Change to 0.1s for faster */
```

---

## 📝 Presentation Tips

### For Your Boss's Review

1. **Print Version:**
   - Open in browser → Print → Save as PDF
   - Both slides fit on one page each
   - Animations won't show in PDF (static view)

2. **Live Presentation:**
   - Use browser in full-screen mode (F11)
   - Animations play automatically on page load
   - Reload page (F5) to replay animations

3. **Talking Points:**
   - Emphasize **physics-informed** approach (not just black-box ML)
   - Highlight **scalability** (800 GPUs)
   - Show **multi-pollutant** capability (6 species simultaneously)
   - Demonstrate **multi-horizon** forecasting (12h to 96h)

### Common Questions to Prepare For

**Q: Why wind-guided scanning?**
A: Atmospheric transport follows wind direction. By processing patches upwind→downwind, the model respects causality in pollutant dispersion.

**Q: Why elevation bias in attention?**
A: Topographic barriers (mountains) block pollutant transport. The bias penalizes uphill attention, encoding physical constraints.

**Q: Why not use a mask [0,1] instead of additive bias?**
A: Multiplicative masks break attention normalization. Additive bias maintains sum=1 through softmax renormalization.

**Q: How do you validate the physics is helping?**
A: Ablation studies comparing: (1) baseline, (2) wind only, (3) elevation only, (4) both innovations.

**Q: Training time seems long (48h) - is this efficient?**
A: For 800 GPUs processing 4+ years of hourly data at 128×256 resolution with 6 pollutants, this is highly efficient. Single-GPU equivalent would be months.

---

## 📚 Related Files

**Documentation:**
- `README.md` - Project overview
- `TOPOFLOW_ATTENTION_EXPLAINED.md` - Deep dive into elevation bias
- `ELEVATION_MASK_EXPERIMENT_SUMMARY.md` - Experimental protocol

**Code:**
- `src/climax_core/parallelpatchembed_wind.py` - Wind scanning implementation
- `src/climax_core/physics_attention_patch_level.py` - Elevation attention
- `src/model_multipollutants.py` - Main model architecture

**Results:**
- `logs/wind_aware_v41_rmse_13073257.out` - Detailed RMSE evaluation
- `logs/multipollutants_climax_ddp/version_47/` - Best checkpoint directory

---

## 🎯 Key Takeaways for Your Boss

1. **Novel Physics Integration:**
   - Wind-guided patch scanning (captures atmospheric transport)
   - Elevation-aware attention (models topographic barriers)

2. **Scalability Demonstrated:**
   - Successfully trained on 800 GPUs (LUMI supercomputer)
   - Handles 6 pollutants × 4 horizons simultaneously

3. **Strong Performance:**
   - Validation loss: 0.3557
   - RMSE metrics competitive across all pollutants and horizons
   - Stable predictions from 12h to 96h

4. **Scientific Rigor:**
   - Physics-informed inductive biases
   - Learnable parameters (α) for physics strength
   - Comprehensive evaluation on held-out test year

---

## 📧 Contact

**Questions or modifications needed?**

Let me know if you need:
- Different color schemes
- Additional slides (architecture details, ablation studies, etc.)
- Export to PowerPoint format
- Interactive features (clickable animations, hover effects)
- Comparison charts (TopoFlow vs. baseline)

---

**Good luck with your presentation!** 🚀
