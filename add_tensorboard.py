# Modifications à faire dans main.py

# 1. Importer TensorBoardLogger
from pytorch_lightning import loggers as pl_loggers

# 2. Remplacer ou compléter votre logger actuel
def setup_loggers(name, version=None):
    """Configure les loggers pour l'entraînement"""
    
    # TensorBoard Logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="logs",
        name=name,
        version=version,
        log_graph=True,  # Log le graphe du modèle
        default_hp_metric=False
    )
    
    # CSV Logger (garder pour compatibilité)
    csv_logger = pl_loggers.CSVLogger(
        save_dir="logs",
        name=name,
        version=version
    )
    
    # Utiliser les deux loggers
    return [tb_logger, csv_logger]

# 3. Dans votre fonction main(), remplacer:
# logger = pl_loggers.CSVLogger(...)
# Par:
logger = setup_loggers("climax_from_scratch", version="tensorboard_run")

# 4. Pour logger des métriques custom dans votre modèle:
# Dans training_step:
self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
self.log('learning_rate', self.optimizers().param_groups[0]['lr'], on_step=True, logger=True)

# Dans validation_step:
self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

# 5. Pour surveiller la mémoire GPU:
if self.global_step % 100 == 0:
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    self.log('gpu_memory_gb', gpu_mem, on_step=True, logger=True)
