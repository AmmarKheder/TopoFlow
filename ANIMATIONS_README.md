# 🎬 TopoFlow Animations GIF - Guide Complet

**Créé le:** 19 Octobre 2025
**Auteur:** Ammar Kheddar
**Projet:** TopoFlow PhD Research

---

## ✅ Animations Créées (3 Fichiers GIF)

### 1. **Wind Scanning Animation** 🌬️
**Fichier:** `wind_scanning_animation.gif`
**Taille:** ~2-3 MB
**Durée:** 7.2 secondes (36 frames @ 5 fps)
**Résolution:** 120 DPI

**Contenu:**
- **Panel Gauche:** Scanning standard (row-major order) - ordre arbitraire
- **Panel Droit:** Scanning guidé par le vent - ordre physique
- **Animation:** Rotation complète du vent (0° → 360°)
- **Visualisation:** Ordre de traitement coloré (bleu foncé → jaune)

**Ce que ça montre:**
- Comment les patches sont réorganisés selon la direction du vent
- La différence entre l'ordre arbitraire et l'ordre physique
- Le vent (flèche rouge) qui tourne sur 360°
- L'ordre de traitement qui s'adapte dynamiquement

**Utilisation:**
- Slide "Méthodologie" de votre présentation
- Expliquer l'innovation #1: Wind-Guided Scanning
- Montrer la capture des patterns de transport atmosphérique

---

### 2. **Elevation Attention Animation** 🏔️
**Fichier:** `elevation_attention_animation.gif`
**Taille:** ~3-4 MB
**Durée:** 10 secondes (20 frames @ 2 fps)
**Résolution:** 120 DPI

**Contenu:**
- **Panel Haut Gauche:** Profil de terrain avec point source mobile
- **Panel Haut Droit:** Poids d'attention avec biais d'élévation
- **Panel Bas:** Formule mathématique et explication

**Animations:**
- Point source (étoile rouge) qui se déplace sur le terrain
- Poids d'attention qui changent selon l'élévation
- Couleurs:
  - Rouge = Uphill (pénalisé)
  - Vert = Downhill (normal)
  - Orange = Flat (neutre)

**Formule affichée:**
```
bias_ij = -α × max(0, (elevation_j - elevation_i) / H_scale)
attention_ij = softmax(Q·K^T / √d + bias_ij)
```

**Ce que ça montre:**
- Comment l'attention est modulée par la différence d'élévation
- Pénalisation du transport uphill (montée)
- Transport downhill normal (descente)
- Application AVANT softmax (additif, pas multiplicatif)

**Utilisation:**
- Slide "Méthodologie" pour innovation #2
- Expliquer le biais d'élévation
- Montrer la physique des barrières topographiques

---

### 3. **Combined Animation** 🎬
**Fichier:** `topoflow_combined_animation.gif`
**Taille:** ~4-5 MB
**Durée:** 8 secondes (24 frames @ 3 fps)
**Résolution:** 150 DPI (haute qualité)

**Contenu:**
- **Panel Gauche:** Wind scanning avec rotation du vent
- **Panel Droit:** Elevation attention avec terrain et flèches

**Ce que ça montre:**
- Les deux innovations travaillant ensemble
- Vue d'ensemble compacte
- Animation synchronisée

**Utilisation:**
- Slide d'introduction ou de conclusion
- Résumé visuel des innovations
- Quick overview pour votre chef

---

## 📖 Comment Utiliser les Animations

### Option 1: Insérer dans PowerPoint (RECOMMANDÉ)

#### Méthode A: Insertion Directe
1. Ouvrir votre `TopoFlow_Presentation.pptx`
2. Aller à la slide "Méthodologie"
3. **Insert** → **Pictures** → Sélectionner le GIF
4. Redimensionner et positionner
5. **IMPORTANT:** En mode présentation, le GIF jouera automatiquement!

#### Méthode B: Remplacer une Image Existante
1. Clic droit sur une image existante
2. **Change Picture** → **From File**
3. Sélectionner votre GIF
4. Le GIF conserve la taille/position de l'image remplacée

**Tips PowerPoint:**
- Les GIFs jouent EN BOUCLE pendant la présentation
- Cliquez pour passer à la slide suivante (le GIF continue)
- Pour arrêter: **Slideshow** → **Set Up Show** → **Loop continuously**

---

### Option 2: Afficher dans un Navigateur

```bash
# Sur LUMI
firefox wind_scanning_animation.gif &
firefox elevation_attention_animation.gif &
firefox topoflow_combined_animation.gif &
```

**Ou copier sur votre machine locale:**
```bash
scp khederam@lumi.csc.fi:/pfs/lustrep1/scratch/project_462000640/ammar/aq_net2/*.gif .
```

Puis double-cliquer ou ouvrir dans navigateur.

---

### Option 3: Convertir en Vidéo MP4 (si besoin)

```bash
# Installer ffmpeg (si pas déjà fait)
# Linux: sudo apt install ffmpeg
# Mac: brew install ffmpeg

# Convertir GIF → MP4
ffmpeg -i wind_scanning_animation.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" wind_scanning.mp4

ffmpeg -i elevation_attention_animation.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" elevation_attention.mp4

ffmpeg -i topoflow_combined_animation.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" topoflow_combined.mp4
```

**Avantages MP4:**
- Taille fichier plus petite
- Meilleure compatibilité vidéo
- Contrôles play/pause dans PowerPoint

---

## 🎯 Suggestions de Présentation

### Slide Méthodologie - Animation Wind Scanning

**Placement suggéré:**
- Coin droit de la slide
- Taille: 50% largeur slide
- À côté du texte explicatif

**Script:**
> "Regardez comment les patches sont réorganisés dynamiquement selon la direction du vent..."
>
> (Montrer l'animation qui tourne)
>
> "À gauche: ordre arbitraire standard. À droite: ordre physique upwind→downwind."
>
> "Cette réorganisation capture les patterns de transport atmosphérique - la pollution suit le vent!"

---

### Slide Méthodologie - Animation Elevation

**Placement suggéré:**
- Centre de la slide
- Taille: 70% largeur
- En-dessous du titre

**Script:**
> "Le biais d'élévation pénalise le transport uphill..."
>
> (Montrer le point source qui bouge)
>
> "Voyez comment les poids d'attention (barres) changent selon l'élévation."
>
> "Rouge = uphill pénalisé, Vert = downhill normal."
>
> "Formule simple: bias = -α × max(0, Δh/H) appliquée AVANT softmax."

---

### Slide d'Introduction - Animation Combinée

**Placement suggéré:**
- Plein écran ou 80% largeur
- Centre de la slide
- Après le titre

**Script:**
> "TopoFlow intègre deux innovations physiques..."
>
> (Montrer l'animation combinée)
>
> "Gauche: scanning guidé par le vent. Droite: attention modulée par l'élévation."
>
> "Ces deux mécanismes travaillent ensemble pour capturer la physique du transport atmosphérique."

---

## 🔧 Paramètres Techniques

### Animation 1: Wind Scanning
```python
- Grid: 8×12 patches (96 total)
- Frames: 36 (rotation 360°)
- FPS: 5 frames/seconde
- Durée: 7.2 secondes
- Colormap: viridis (bleu→jaune)
- Angle step: 10° par frame
```

### Animation 2: Elevation Attention
```python
- Terrain: Fonction sinusoïdale (simule montagnes)
- Points: 20 échantillons
- Frames: 20 (source mobile)
- FPS: 2 frames/seconde
- Durée: 10 secondes
- α (alpha): 2.0
- H_scale: 1000 mètres
```

### Animation 3: Combined
```python
- Grid: 6×8 patches (48 total)
- Frames: 24
- FPS: 3 frames/seconde
- Durée: 8 secondes
- Resolution: 150 DPI (haute qualité)
```

---

## 💡 Personnalisation

### Modifier les Couleurs
Éditer `create_animations.py` ligne ~80-90:
```python
# Changer colormap
cmap = plt.cm.viridis  # Options: plasma, inferno, magma, turbo
```

### Changer la Vitesse
```python
# Dans create_wind_scanning_animation()
interval=200  # Modifier cette valeur (ms entre frames)
              # 200ms = lent, 100ms = rapide

# Dans create_elevation_attention_animation()
interval=500  # 500ms = très lent, 200ms = normal
```

### Modifier la Résolution
```python
# Dans save() calls
dpi=120  # Changer à 150 (haute qual) ou 90 (petite taille)
```

### Régénérer les Animations
```bash
cd /scratch/project_462000640/ammar/aq_net2
source venv_pytorch_rocm/bin/activate
python create_animations.py
```

---

## 📊 Comparaison des Fichiers

| Animation | Taille | Durée | Frames | FPS | Qualité | Usage Recommandé |
|-----------|--------|-------|--------|-----|---------|------------------|
| **Wind Scanning** | ~2.5 MB | 7.2s | 36 | 5 | Moyenne | Slide Méthodologie |
| **Elevation Attention** | ~3.8 MB | 10s | 20 | 2 | Moyenne | Slide Méthodologie |
| **Combined** | ~4.2 MB | 8s | 24 | 3 | Haute | Slide Introduction |

---

## ✅ Checklist d'Intégration PowerPoint

- [ ] Copier les 3 GIFs sur votre machine locale
- [ ] Ouvrir `TopoFlow_Presentation.pptx`
- [ ] Insérer `wind_scanning_animation.gif` dans slide Méthodologie
- [ ] Insérer `elevation_attention_animation.gif` dans slide Méthodologie
- [ ] (Optionnel) Insérer `topoflow_combined_animation.gif` dans slide Introduction
- [ ] Redimensionner les GIFs (50-70% largeur slide)
- [ ] Tester en mode Slideshow (F5)
- [ ] Vérifier que les GIFs jouent automatiquement
- [ ] Ajuster timing si nécessaire (clic pour passer)

---

## 🎨 Design Tips

### Placement Optimal
```
┌─────────────────────────────────────┐
│  TITLE: Wind-Guided Scanning        │
├──────────────────┬──────────────────┤
│                  │                  │
│  TEXTE           │   GIF ANIMATION  │
│  EXPLICATIF      │   (50% largeur)  │
│                  │                  │
│  • Point 1       │                  │
│  • Point 2       │                  │
│  • Point 3       │                  │
└──────────────────┴──────────────────┘
```

### Transitions PowerPoint
- Utilisez **Fade** ou **Push** pour transitions fluides
- Évitez transitions trop flashy (distrait de l'animation)
- Timing: 0.5-1 seconde par transition

### Texte & Animation
- Placez le texte À GAUCHE
- Animation À DROITE
- Laissez espace blanc (breathing room)
- Police: 18-24pt pour texte explicatif

---

## 🔍 Troubleshooting

### Le GIF ne joue pas dans PowerPoint
**Solution:**
1. Vérifier version PowerPoint (2010+ requis)
2. En mode édition: le GIF est statique (NORMAL)
3. En mode présentation (F5): le GIF joue
4. Si problème persiste: convertir en MP4 (voir Option 3)

### Le GIF est trop grand/petit
**Solution:**
1. Clic droit → **Size and Position**
2. Décocher **Lock aspect ratio** si besoin
3. Ajuster largeur/hauteur
4. Ou: Éditer `create_animations.py` et changer `figsize=(16, 7)`

### Le GIF est flou
**Solution:**
1. Augmenter DPI dans `create_animations.py`:
   ```python
   anim.save(output_file, writer=writer, dpi=150)  # Au lieu de 120
   ```
2. Régénérer: `python create_animations.py`

### Le GIF lag/saccade
**Solution:**
1. Réduire FPS ou nombre de frames
2. Ou: Convertir en MP4 (plus fluide)
3. Ou: Réduire résolution (DPI=90)

---

## 📧 Support

**Questions ou modifications?**
- Script source: `create_animations.py`
- Documention: Cette page
- Contact: khederam@lumi.csc.fi

---

## 🎓 Explication Technique

### Wind Scanning Animation
**Algorithme:**
```python
for frame in range(36):
    angle = frame * 10°  # 0° to 360°

    for each patch (i, j):
        # Project onto wind direction
        projection = i * cos(angle) + j * sin(angle)

    # Sort patches by projection
    sorted_patches = sort(patches, key=projection)

    # Color by new order
    color = colormap(new_index / total_patches)
```

**Pourquoi ça marche:**
- Projection = distance le long du vecteur vent
- Ordre croissant = upwind → downwind
- Transformers traitent dans cet ordre séquentiel

---

### Elevation Attention Animation
**Algorithme:**
```python
for frame in range(20):
    source_idx = frame
    source_elevation = terrain[source_idx]

    for target in all_targets:
        # Compute elevation bias
        delta_h = target_elevation - source_elevation
        bias = -alpha * max(0, delta_h / H_scale)
        bias = clamp(bias, -10, 0)

        # Attention weight (simplified)
        attention = exp(base_score + bias)

    # Normalize (softmax)
    attention_weights = normalize(attention)
```

**Physique:**
- delta_h > 0: Uphill → bias négatif → attention réduite
- delta_h < 0: Downhill → bias = 0 → attention normale
- Softmax renormalise pour sum = 1

---

**Bon succès avec vos animations! 🎬🚀**
