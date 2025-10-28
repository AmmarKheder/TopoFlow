# 📑 INDEX COMPLET - Présentations TopoFlow

**Date de création:** 19 Octobre 2025
**Auteur:** Ammar Kheddar
**Projet:** TopoFlow - PhD Research in AI for Air Quality

---

## 🎯 Vue d'Ensemble Rapide

Vous avez maintenant **TOUT** ce qu'il faut pour présenter TopoFlow à votre chef:

✅ **1 PowerPoint professionnel** (6 slides, graphiques, cartes)
✅ **2 présentations HTML** (animations CSS interactives)
✅ **3 animations GIF** (wind scanning + elevation attention)
✅ **5 guides complets** (README, explications, scripts)

**Temps total de création:** ~2 heures
**Fichiers générés:** 11 fichiers

---

## 📂 Tous les Fichiers Créés

### 🎯 FICHIER PRINCIPAL (À PRÉSENTER)

#### **TopoFlow_Presentation.pptx** ⭐⭐⭐ PRIORITÉ #1
- **Format:** Microsoft PowerPoint
- **Taille:** 1.1 MB
- **Slides:** 6 diapositives professionnelles
- **Contenu:**
  1. Page de titre
  2. Méthodologie (architecture + 4 innovations)
  3. Résultats - Graphique RMSE
  4. Résultats - Évolution MAE
  5. Tableau comparatif détaillé
  6. Carte Chine PM2.5

**📍 Location:**
```
/scratch/project_462000640/ammar/aq_net2/TopoFlow_Presentation.pptx
```

**📥 Comment télécharger:**
```bash
scp khederam@lumi.csc.fi:/pfs/lustrep1/scratch/project_462000640/ammar/aq_net2/TopoFlow_Presentation.pptx .
```

---

### 🎬 ANIMATIONS GIF (À INSÉRER DANS POWERPOINT)

#### 1. **wind_scanning_animation.gif** 🌬️
- **Taille:** 3.6 MB
- **Durée:** 7.2 secondes (36 frames)
- **Montre:** Comment patches sont réorganisés par direction du vent
- **Usage:** Slide Méthodologie - Innovation #1

#### 2. **elevation_attention_animation.gif** 🏔️
- **Taille:** 1.2 MB
- **Durée:** 10 secondes (20 frames)
- **Montre:** Comment attention est modulée par élévation terrain
- **Usage:** Slide Méthodologie - Innovation #2

#### 3. **topoflow_combined_animation.gif** 🎬
- **Taille:** 2.0 MB
- **Durée:** 8 secondes (24 frames)
- **Montre:** Les deux innovations ensemble
- **Usage:** Slide Introduction ou Conclusion

**📍 Location:** Même répertoire que le PPTX

**📥 Télécharger tous les GIFs:**
```bash
scp khederam@lumi.csc.fi:/pfs/lustrep1/scratch/project_462000640/ammar/aq_net2/*.gif .
```

---

### 🌐 PRÉSENTATIONS HTML (ALTERNATIVES)

#### **presentation_methodology.html**
- Format web interactif avec animations CSS
- Thème: Gradient violet/bleu
- Contenu: Méthodologie détaillée
- Auto-play des animations

#### **presentation_results.html**
- Format web interactif
- Thème: Gradient vert/teal
- Contenu: Résultats + carte Chine animée
- Carte SVG qui pulse

**📖 Usage:**
- Double-cliquer pour ouvrir dans navigateur
- Ou: Convertir en PDF (Ctrl+P → Save as PDF)

---

### 📚 GUIDES & DOCUMENTATION

#### 1. **PRESENTATION_README.md** (13 KB)
Guide complet avec:
- Instructions détaillées
- Scripts de présentation (8-10 minutes)
- Questions anticipées + réponses
- Tips PowerPoint

#### 2. **SUMMARY_FOR_BOSS.md** (12 KB)
Résumé exécutif avec:
- Vue d'ensemble 30 secondes
- Innovations clés
- Résultats principaux
- Points à mentionner

#### 3. **ANIMATIONS_README.md** (10 KB)
Guide des animations avec:
- Description de chaque GIF
- Instructions d'insertion PowerPoint
- Troubleshooting
- Personnalisation

#### 4. **PRESENTATION_GUIDE.md** (15 KB)
Guide original des présentations HTML

#### 5. **INDEX_PRESENTATIONS.md** (ce fichier)
Index complet de tout

---

## 🚀 Quick Start Guide (5 minutes)

### Étape 1: Télécharger les Fichiers Essentiels
```bash
# Sur votre machine locale
cd ~/Desktop  # ou votre dossier préféré

# Télécharger PowerPoint
scp khederam@lumi.csc.fi:/pfs/lustrep1/scratch/project_462000640/ammar/aq_net2/TopoFlow_Presentation.pptx .

# Télécharger animations
scp khederam@lumi.csc.fi:/pfs/lustrep1/scratch/project_462000640/ammar/aq_net2/*.gif .

# Télécharger guides (optionnel)
scp khederam@lumi.csc.fi:/pfs/lustrep1/scratch/project_462000640/ammar/aq_net2/*README.md .
```

### Étape 2: Ouvrir PowerPoint
```bash
# Windows
start TopoFlow_Presentation.pptx

# Mac
open TopoFlow_Presentation.pptx

# Linux
libreoffice TopoFlow_Presentation.pptx
```

### Étape 3: Insérer les Animations (Optionnel mais Recommandé)
1. Aller à Slide 2 (Méthodologie)
2. **Insert** → **Pictures** → Sélectionner `wind_scanning_animation.gif`
3. Redimensionner et placer à droite du texte
4. Répéter avec `elevation_attention_animation.gif`
5. Tester en mode Slideshow (F5)

### Étape 4: Préparer Votre Présentation
1. Lire `SUMMARY_FOR_BOSS.md` (5 minutes)
2. Parcourir `PRESENTATION_README.md` section "Script de Présentation"
3. Noter les points clés sur papier/notes
4. Chronométrer votre présentation (objectif: 8-10 minutes)

---

## 📊 Structure de Présentation Suggérée

### Slide 1: Titre (30 secondes)
> "Bonjour, aujourd'hui je présente TopoFlow, une architecture de deep learning guidée par la physique pour la prévision multi-polluants de qualité de l'air."

### Slide 2: Méthodologie (4 minutes)
**Innovation #1: Wind Scanning** (1.5 min)
- Montrer animation GIF
- Expliquer réorganisation upwind→downwind
- 16 secteurs pré-calculés

**Innovation #2: Elevation Attention** (2 min)
- Montrer animation GIF
- Expliquer biais d'élévation
- Formule: bias = -α × max(0, Δh/H)
- Appliqué AVANT softmax

**Infrastructure** (30 sec)
- 800 GPUs AMD MI250X
- 48 heures training
- PyTorch Lightning DDP

### Slides 3-5: Résultats (4 minutes)
**Slide 3: RMSE Chart** (1.5 min)
- 6 polluants × 4 horizons
- SO₂ meilleure stabilité
- PM2.5 @ 24h = 12.67 µg/m³

**Slide 4: MAE Evolution** (1.5 min)
- Trends par polluant
- O₃ complexité photochimique
- Multi-horizon stable

**Slide 5: Table** (1 min)
- Récapitulatif complet
- Val loss: 0.3557
- 11,317 pixels actifs

### Slide 6: Carte Chine (1 minute)
> "Visualisation spatiale montrant distribution réaliste avec hotspots urbains capturés."

### Conclusion (30 secondes)
> "TopoFlow démontre que les biais inductifs guidés par la physique - scanning par vent et attention par élévation - permettent une prévision précise et scalable multi-polluants."

**Total:** 8-10 minutes

---

## 🎯 Checklist Complète Avant Présentation

### Préparation Fichiers
- [ ] PowerPoint téléchargé sur machine locale
- [ ] Animations GIF téléchargées
- [ ] PowerPoint s'ouvre correctement
- [ ] Toutes images/graphiques visibles
- [ ] (Optionnel) GIFs insérés dans slides

### Préparation Contenu
- [ ] Lu SUMMARY_FOR_BOSS.md
- [ ] Lu PRESENTATION_README.md section Scripts
- [ ] Préparé réponses aux questions anticipées
- [ ] Chronométré présentation (8-10 min)
- [ ] Noté points clés sur papier

### Test Technique
- [ ] Testé mode Slideshow (F5 dans PowerPoint)
- [ ] Vérifié transitions fonctionnent
- [ ] Testé GIFs jouent automatiquement (si insérés)
- [ ] Backup PDF créé (Ctrl+P → Save as PDF)

### Données de Référence
- [ ] Checkpoint: `version_47, step 311, val_loss=0.3557`
- [ ] Test year: 2018
- [ ] Coverage: 11,317 pixels (34.5%)
- [ ] Best RMSE: SO₂ @ 48h = 2.81 µg/m³

---

## 💡 Points Clés à Retenir

### Les 3 Messages Principaux

1. **Physics-Informed Design** 🔬
   - Pas juste un modèle black-box
   - Intègre connaissance du domaine (météo, topographie)
   - Innovations explicables et interprétables

2. **Multi-Pollutant + Multi-Horizon** 🧪
   - 6 espèces simultanément (vs. 1 dans littérature)
   - 12h à 96h (4 jours ahead)
   - Performance stable sur tous horizons

3. **Large-Scale Validation** ⚡
   - 800 GPUs (parmi plus grandes études ML air quality)
   - Données réelles 2013-2018
   - Résultats compétitifs (val_loss 0.3557)

---

## 📈 Métriques Clés à Mentionner

### Infrastructure
- **GPUs:** 800 AMD MI250X
- **Nodes:** 100 × 8 GPUs/node
- **Training:** ~48 heures
- **Framework:** PyTorch Lightning DDP

### Données
- **Train:** 2013-2016 (4 ans, horaire)
- **Validation:** 2017 (1 an)
- **Test:** 2018 (1 an)
- **Resolution:** 128×256 (Chine + Taiwan)
- **Coverage:** 34.5% (11,317 pixels actifs)

### Performance (Horizon 24h)
- **PM2.5:** RMSE=12.67, MAE=5.46 µg/m³
- **SO₂:** RMSE=2.88, MAE=1.41 µg/m³ ⭐
- **NO₂:** RMSE=9.16, MAE=4.24 µg/m³
- **O₃:** RMSE=19.97, MAE=14.07 µg/m³

---

## 🔗 Liens Utiles

### Fichiers sur LUMI
```
/scratch/project_462000640/ammar/aq_net2/
├── TopoFlow_Presentation.pptx          (PowerPoint principal)
├── wind_scanning_animation.gif          (Animation vent)
├── elevation_attention_animation.gif    (Animation élévation)
├── topoflow_combined_animation.gif      (Animation combinée)
├── presentation_methodology.html        (HTML méthodologie)
├── presentation_results.html            (HTML résultats)
├── PRESENTATION_README.md               (Guide principal)
├── SUMMARY_FOR_BOSS.md                  (Résumé exécutif)
├── ANIMATIONS_README.md                 (Guide animations)
├── PRESENTATION_GUIDE.md                (Guide HTML)
└── INDEX_PRESENTATIONS.md               (Ce fichier)
```

### Données Sources
```
- Métriques: archive/results_old/eval_baseline_20250923_024726/baseline_metrics.json
- Cartes: archive/media/map_*.png
- Checkpoint: logs/multipollutants_climax_ddp/version_47/checkpoints/
- Config: configs/config_all_pollutants.yaml
```

### Code Source
```
- Architecture: src/climax_core/arch.py
- Wind scanning: src/climax_core/parallelpatchembed_wind.py
- Elevation attention: src/climax_core/physics_attention_patch_level.py
- Model: src/model_multipollutants.py
```

---

## 🛠️ Scripts de Génération (Pour Régénérer si Besoin)

### Régénérer PowerPoint
```bash
cd /scratch/project_462000640/ammar/aq_net2
source venv_pytorch_rocm/bin/activate
python create_powerpoint.py
```

### Régénérer Animations
```bash
cd /scratch/project_462000640/ammar/aq_net2
source venv_pytorch_rocm/bin/activate
python create_animations.py
```

### Les HTML sont statiques (pas besoin de régénérer)

---

## ❓ FAQ Rapide

**Q: Quel fichier envoyer à mon chef?**
**R:** `TopoFlow_Presentation.pptx` + les 3 GIFs (si vous les insérez)

**Q: Combien de temps pour la présentation?**
**R:** 8-10 minutes (6 slides, ~1.5 min/slide)

**Q: Les GIFs sont obligatoires?**
**R:** Non, mais FORTEMENT recommandés. Rendent la présentation beaucoup plus claire.

**Q: Comment insérer GIFs dans PowerPoint?**
**R:** Insert → Pictures → Sélectionner le GIF. En mode Slideshow, il jouera automatiquement.

**Q: Que faire si PowerPoint crashe?**
**R:** Créer un PDF backup: Ouvrir PPTX → Ctrl+P → Save as PDF

**Q: Les animations HTML sont-elles nécessaires?**
**R:** Non, elles sont une alternative. Le PowerPoint est suffisant.

**Q: Puis-je modifier les slides?**
**R:** Oui! Le PPTX est éditable. Ajoutez votre logo, changez couleurs, etc.

---

## 🎓 Ressources Supplémentaires

### Documentation Technique
- `README.md` - Vue d'ensemble projet TopoFlow
- `TOPOFLOW_ATTENTION_EXPLAINED.md` - Deep dive mathématique
- `ELEVATION_MASK_EXPERIMENT_SUMMARY.md` - Protocole expérimental

### Papiers & References
- ClimaX: Foundation Model for Weather and Climate
- Attention is All You Need (Transformer architecture)
- ALiBi: Attention with Linear Biases

---

## ✅ Résumé Final

**Vous avez maintenant:**
✅ 1 PowerPoint pro avec 6 slides + graphiques + cartes
✅ 3 animations GIF expliquant les innovations
✅ 2 présentations HTML interactives (alternatives)
✅ 5 guides complets (README, scripts, FAQ)

**Ce que vous devez faire:**
1. ⬇️ Télécharger `TopoFlow_Presentation.pptx`
2. ⬇️ Télécharger les 3 GIFs
3. 📖 Lire `SUMMARY_FOR_BOSS.md` (5 min)
4. ✏️ Insérer GIFs dans PPTX (10 min)
5. 🎤 Pratiquer présentation (2-3 fois)

**Temps total:** ~30 minutes de préparation
**Résultat:** Présentation professionnelle prête! 🚀

---

## 📧 Contact & Support

**Questions?**
- Email: khederam@lumi.csc.fi
- Projet: TopoFlow PhD Research
- Institution: [Votre université]

**Modifications nécessaires?**
- Couleurs, logo, texte → Éditer PPTX directement
- Graphiques, data → Éditer `create_powerpoint.py` puis régénérer
- Animations → Éditer `create_animations.py` puis régénérer

---

# 🎉 Bonne Présentation! 🚀🎓

**Tous les fichiers sont prêts dans:**
```
/scratch/project_462000640/ammar/aq_net2/
```

**N'oubliez pas:**
- Soyez confiant (vous avez d'excellents résultats!)
- Expliquez la physique (c'est votre force unique)
- Montrez les animations (valent 1000 mots)
- Répondez aux questions calmement (vous connaissez le sujet!)

**Votre message clé:**
> "TopoFlow démontre que l'intégration de connaissances physiques - via scanning guidé par le vent et attention modulée par l'élévation - permet une prévision précise et scalable de la qualité de l'air multi-polluants."

✨ **Vous êtes prêt!** ✨
