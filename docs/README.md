# TopoFlow GitHub Pages

This directory contains the GitHub Pages website for TopoFlow.

## Deployment Instructions

### 1. Push to GitHub

```bash
# From project root
git add docs/
git commit -m "Add TopoFlow GitHub Pages website"
git push origin main
```

### 2. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under "Source", select:
   - **Branch:** `main`
   - **Folder:** `/docs`
4. Click **Save**

Your site will be live at: `https://yourusername.github.io/topoflow/`

## Local Development

To preview the website locally:

```bash
# Option 1: Python simple HTTP server
cd docs
python3 -m http.server 8000

# Option 2: Using Ruby (if available)
cd docs
ruby -run -ehttpd . -p8000
```

Then open: http://localhost:8000

## Updating Visualizations

To regenerate the visualizations:

```bash
cd scripts
python generate_website_visuals.py
```

This will update all images and animations in `docs/assets/`.

## File Structure

```
docs/
├── index.html          # Main page
├── style.css           # Styling
├── script.js           # Animations and interactions
├── assets/             # Generated visualizations
│   ├── wind_reorder_demo.gif
│   ├── regional_grid_32x32.png
│   ├── elevation_bias_demo.png
│   ├── architecture.png
│   ├── prediction_pm25.png
│   └── prediction_no2.png
└── README.md           # This file
```

## Customization

### Update Results

Edit `index.html` and modify the results section (around line 200) with your actual validation metrics.

### Change Colors

Edit `style.css` and modify the `:root` CSS variables:

```css
:root {
    --primary-color: #2563eb;    /* Blue */
    --secondary-color: #7c3aed;  /* Purple */
    --dark: #1e293b;             /* Dark gray */
}
```

### Add More Visualizations

1. Generate new images and save to `docs/assets/`
2. Reference them in `index.html`:

```html
<img src="assets/your_new_image.png" alt="Description">
```

## Tips

- **Performance:** Optimize GIF size with tools like `gifsicle`:
  ```bash
  gifsicle -O3 --colors 256 input.gif -o output.gif
  ```

- **Images:** Compress PNG images with `optipng`:
  ```bash
  optipng -o7 image.png
  ```

- **Mobile:** Test on mobile by using browser dev tools (F12 → Toggle device toolbar)

## License

MIT License - Feel free to use this template for your own projects!
