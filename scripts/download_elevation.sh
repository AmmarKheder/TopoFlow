#!/bin/bash
#SBATCH --job-name=download_elevation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --partition=small-g
#SBATCH --account=project_462000640
#SBATCH --output=logs/download_elevation_%j.out
#SBATCH --error=logs/download_elevation_%j.err

echo "🌍 Téléchargement des données d'élévation SRTM/GEBCO pour la Chine"
echo "Région: 78°E-134°E, 10°N-53°N (grille 339x432)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "======================================================"

cd /scratch/project_462000640/ammar/aq_net2

# Créer le dossier de données
mkdir -p elevation_data
cd elevation_data

echo "📥 Étape 1: Téléchargement GEBCO (bathymétrie + topographie mondiale)"
echo "Source: General Bathymetric Chart of the Oceans"

# GEBCO 2023 - Dataset complet mondial (recommandé)
echo "Téléchargement GEBCO 2023 grille complète..."
wget -c -O gebco_2023_global.nc.zip \
  "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2023/zip/GEBCO_2023.nc.zip"

if [ -f "gebco_2023_global.nc.zip" ]; then
    echo "✅ GEBCO global téléchargé (taille: $(du -h gebco_2023_global.nc.zip | cut -f1))"
    unzip -o gebco_2023_global.nc.zip
    rm gebco_2023_global.nc.zip
else
    echo "❌ Échec téléchargement GEBCO global"
fi

echo ""
echo "📥 Étape 2: Téléchargement SRTM tuiles individuelles"
echo "Source: Shuttle Radar Topography Mission (NASA)"

# Liste des tuiles SRTM nécessaires pour couvrir la Chine
# Basé sur la grille lat 10-53°N, lon 78-134°E
declare -a srtm_tiles=(
    # Sud de la Chine (10-20°N)
    "N10E078" "N10E079" "N10E080" "N10E081" "N10E082" "N10E083" "N10E084" "N10E085"
    "N11E090" "N11E091" "N11E092" "N11E093" "N11E094" "N11E095" "N11E096" "N11E097"
    "N12E100" "N12E101" "N12E102" "N12E103" "N12E104" "N12E105" "N12E106" "N12E107"
    
    # Centre de la Chine (20-35°N)  
    "N20E095" "N20E096" "N20E097" "N20E098" "N20E099" "N20E100" "N20E101" "N20E102"
    "N25E105" "N25E106" "N25E107" "N25E108" "N25E109" "N25E110" "N25E111" "N25E112"
    "N30E110" "N30E111" "N30E112" "N30E113" "N30E114" "N30E115" "N30E116" "N30E117"
    
    # Nord de la Chine (35-45°N)
    "N35E115" "N35E116" "N35E117" "N35E118" "N35E119" "N35E120" "N35E121" "N35E122"
    "N40E125" "N40E126" "N40E127" "N40E128" "N40E129" "N40E130" "N40E131" "N40E132"
    
    # Extrême nord (45-53°N)
    "N45E130" "N45E131" "N45E132" "N45E133" "N45E134"
    "N50E131" "N50E132" "N50E133" "N50E134"
    "N52E131" "N52E132" "N52E133"
)

echo "Téléchargement de ${#srtm_tiles[@]} tuiles SRTM..."
mkdir -p srtm_tiles

# Téléchargement parallèle des tuiles SRTM
download_count=0
for tile in "${srtm_tiles[@]}"; do
    echo "[$((++download_count))/${#srtm_tiles[@]}] Téléchargement $tile..."
    
    # Essayer plusieurs sources pour SRTM
    success=false
    
    # Source 1: USGS EarthExplorer (principal)
    if [ "$success" = false ]; then
        wget -c -q -O "srtm_tiles/${tile}.hgt.zip" \
          "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/${tile}.SRTMGL1.hgt.zip" \
          && success=true
    fi
    
    # Source 2: OpenTopography (backup)  
    if [ "$success" = false ]; then
        wget -c -q -O "srtm_tiles/${tile}.hgt.zip" \
          "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1/${tile}.SRTMGL1.hgt.zip" \
          && success=true
    fi
    
    # Source 3: ViewfinderPanoramas (backup alternatif)
    if [ "$success" = false ]; then
        lat_num=${tile:1:2}
        lon_num=${tile:4:3}
        wget -c -q -O "srtm_tiles/${tile}.hgt.zip" \
          "http://viewfinderpanoramas.org/dem1/${tile}.zip" \
          && success=true
    fi
    
    if [ "$success" = true ] && [ -f "srtm_tiles/${tile}.hgt.zip" ]; then
        # Décompresser
        unzip -o "srtm_tiles/${tile}.hgt.zip" -d srtm_tiles/ 2>/dev/null
        rm "srtm_tiles/${tile}.hgt.zip"
        echo "  ✅ $tile téléchargé"
    else
        echo "  ❌ $tile échec"
    fi
    
    # Pause pour éviter de surcharger les serveurs
    sleep 1
done

echo ""
echo "📊 Résumé des téléchargements:"
if [ -f "GEBCO_2023.nc" ]; then
    echo "✅ GEBCO global: $(du -h GEBCO_2023.nc | cut -f1)"
fi

hgt_count=$(ls srtm_tiles/*.hgt 2>/dev/null | wc -l)
echo "✅ Tuiles SRTM: $hgt_count fichiers .hgt"

total_size=$(du -sh . | cut -f1)
echo "📁 Taille totale: $total_size"

echo ""
echo "🔍 Vérification des fichiers téléchargés:"
echo "GEBCO NetCDF:"
ls -la *.nc 2>/dev/null || echo "  Aucun fichier NetCDF"

echo "SRTM HGT:"
ls -la srtm_tiles/*.hgt 2>/dev/null | head -5
if [ $hgt_count -gt 5 ]; then
    echo "  ... et $((hgt_count - 5)) autres fichiers HGT"
fi

echo ""
echo "✅ Téléchargement terminé!"
echo "Fichiers sauvegardés dans: $(pwd)"

# Revenir au répertoire principal
cd /scratch/project_462000640/ammar/aq_net2

echo ""
echo "📋 Prochaines étapes:"
echo "1. Traiter les données: python process_elevation_to_zarr.py"
echo "2. Vérifier l'ajout aux fichiers zarr"
echo "3. Tester l'ordre topographique dans le modèle"

echo ""
echo "🏁 Job terminé: $(date)"
