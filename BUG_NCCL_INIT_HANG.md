# BUG CRITIQUE : NCCL Init Hang après commit fd3c61a5

**Date :** 28 octobre 2025
**Auteur du diagnostic :** Claude (assistant IA)

## Problème

Tous les jobs lancés après le commit `fd3c61a5` (28/10/2025 15:06:21) se bloquent indéfiniment pendant l'initialisation NCCL/RCCL. Les processus distribués s'enregistrent correctement (128/128 ou 256/256) mais l'init NCCL ne se termine jamais.

### Symptômes

- Les 128 processus PyTorch Lightning s'enregistrent : ✅
  ```
  All distributed processes registered. Starting with 128 processes
  ```
- Sortie NCCL s'arrête après :
  ```
  RCCL version 2.18.3+hip6.0 HEAD:2f6d59e+
  ```
- Aucun message suivant (`Setting hipLimitStackSize`, `NET/Socket`, channels config, etc.)
- Le job reste bloqué indéfiniment (testé jusqu'à 26+ minutes)

### Timeline des événements

| Heure | Événement | Statut |
|-------|-----------|--------|
| 14:32:57 | Commit `28d088b4` - Wind Scanning config | ✅ OK |
| 15:00:13 | Commit `a54c0475` - TopoFlow Block 0 | ✅ OK |
| 15:00:36 | **Job 14029003** START (256 GPUs, 32 nodes) | ✅ **Init NCCL réussie** |
| 15:03:48 | Job 14029003 crash (erreur checkpoint) | ⚠️ Crash après init |
| 15:06:21 | **Commit `fd3c61a5`** - Fix MLP structure | ❌ **COMMIT PROBLÉMATIQUE** |
| 15:06:29 | Job 14029179 START (256 GPUs, 32 nodes) | ❌ **Bloqué à init NCCL** |
| 15:35:00 | Job 14029179 annulé (28 min bloqué) | ❌ |
| 15:39:00 | Job 14030236 START (128 GPUs, 16 nodes) | ❌ **Bloqué à init NCCL** |
| 16:05:00 | Job 14030236 annulé (26 min bloqué) | ❌ |
| 16:07:00 | Job 14031138 START (128 GPUs, checkpoint désactivé) | ❌ **Toujours bloqué !** |

## Commit problématique

**Commit :** `fd3c61a5e119d57dbe26a3f5a9d22d5847d08212`
**Date :** 2025-10-28 15:06:21 +0200
**Message :** "Fix MLP structure in PhysicsGuidedBlock for checkpoint compatibility"

### Changements introduits

```python
# AVANT (nn.Sequential)
self.mlp = nn.Sequential(
    nn.Linear(dim, mlp_hidden_dim),
    nn.GELU(),
    nn.Dropout(drop),
    nn.Linear(mlp_hidden_dim, dim),
    nn.Dropout(drop),
)

# APRÈS (classe Mlp personnalisée)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        # ... avec fc1/fc2 au lieu de 0/3

self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
```

**Fichier modifié :** `src/climax_core/topoflow.py`

## Analyse technique

### Ce qui fonctionne
- Job 14029003 avec commit `a54c0475` :
  - Init NCCL complète en quelques secondes
  - Affiche tous les messages NCCL (channels, trees, P2P, etc.)
  - Crash ensuite à cause d'incompatibilité checkpoint (clés manquantes)

### Ce qui ne fonctionne pas
- Tous les jobs avec commit `fd3c61a5` ou plus récent :
  - Bloqués AVANT que NCCL ne crée le thread de communication
  - Aucun message de channels/trees/P2P
  - Testé avec 256 GPUs, 128 GPUs, avec/sans checkpoint : **même résultat**

### Hypothèse sur la cause

Le changement de structure MLP pourrait causer un problème lors de :
1. L'initialisation du modèle sur chaque rank
2. Une synchronisation implicite early dans PyTorch Lightning
3. Un deadlock lors du premier `allreduce` de paramètres du modèle

Le problème survient AVANT même le chargement du checkpoint, car désactiver le checkpoint ne résout pas le problème.

## Solution temporaire

**Revert au commit `a54c0475` (ou `28d088b4` si besoin) :**

```bash
cd /scratch/project_462000640/ammar/aq_net2
git checkout a54c0475
# Ou pour être plus sûr :
git checkout 28d088b4
```

⚠️ **Attention :** Cela restaurera le code AVANT la correction du checkpoint, donc on aura à nouveau l'erreur de clés manquantes.

## Solution permanente (TODO)

1. Identifier pourquoi la classe `Mlp` cause un deadlock NCCL
2. Options :
   - Garder `nn.Sequential` et corriger l'incompatibilité checkpoint autrement
   - Débugger la classe `Mlp` pour comprendre le problème
   - Utiliser `strict=False` lors du chargement du checkpoint
   - Reconstruire un nouveau checkpoint compatible

## Tests effectués

- ✅ Réduction nœuds : 32 → 16 (pas d'effet)
- ✅ Désactivation checkpoint (pas d'effet)
- ❌ Revert commit (pas encore testé, à faire maintenant)

## Jobs affectés

- 14029179 (256 GPUs, 32 nodes) - Bloqué 28 min
- 14030236 (128 GPUs, 16 nodes) - Bloqué 26 min
- 14031138 (128 GPUs, 16 nodes, no checkpoint) - Bloqué 14+ min

## Commande pour reproduire

```bash
cd /scratch/project_462000640/ammar/aq_net2
git checkout fd3c61a5
sbatch submit_multipollutants_from_6pollutants.sh
# → Job se bloque à l'init NCCL
```

## Notes

- Le problème n'est PAS lié au nombre de nœuds/GPUs
- Le problème n'est PAS lié au checkpoint
- Le problème est SPÉCIFIQUE au changement de code dans le commit fd3c61a5
- Le job 14029003 prouve que le code d'AVANT fonctionnait
