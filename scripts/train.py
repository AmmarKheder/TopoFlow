import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from topoflow import TopoFlowLightningModule, TopoFlowDataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    datamodule = TopoFlowDataModule(config)

    if args.checkpoint:
        model = TopoFlowLightningModule.load_from_checkpoint(
            args.checkpoint, config=config
        )
    else:
        model = TopoFlowLightningModule(config)

    callbacks = []

    if "callbacks" in config:
        if "model_checkpoint" in config["callbacks"]:
            ckpt_cfg = config["callbacks"]["model_checkpoint"]
            callbacks.append(
                ModelCheckpoint(
                    monitor=ckpt_cfg.get("monitor", "val_loss"),
                    mode=ckpt_cfg.get("mode", "min"),
                    save_top_k=ckpt_cfg.get("save_top_k", 3),
                    filename="best-{val_loss:.4f}-{epoch}",
                )
            )

        if "early_stopping" in config["callbacks"]:
            es_cfg = config["callbacks"]["early_stopping"]
            callbacks.append(
                EarlyStopping(
                    monitor=es_cfg.get("monitor", "val_loss"),
                    patience=es_cfg.get("patience", 10),
                    mode=es_cfg.get("mode", "min"),
                )
            )

    train_cfg = config["train"]
    trainer = pl.Trainer(
        max_epochs=train_cfg.get("epochs", 100),
        devices=train_cfg.get("devices", 1),
        num_nodes=train_cfg.get("num_nodes", 1),
        accelerator=train_cfg.get("accelerator", "auto"),
        strategy=train_cfg.get("strategy", "auto"),
        precision=train_cfg.get("precision", 32),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
