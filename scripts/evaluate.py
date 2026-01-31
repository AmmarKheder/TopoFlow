import argparse
import yaml
import pytorch_lightning as pl

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from topoflow import TopoFlowLightningModule, TopoFlowDataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    datamodule = TopoFlowDataModule(config)
    model = TopoFlowLightningModule.load_from_checkpoint(
        args.checkpoint, config=config
    )

    train_cfg = config["train"]
    trainer = pl.Trainer(
        devices=train_cfg.get("devices", 1),
        num_nodes=train_cfg.get("num_nodes", 1),
        accelerator=train_cfg.get("accelerator", "auto"),
        strategy=train_cfg.get("strategy", "auto"),
    )

    results = trainer.test(model, datamodule)
    print("\nTest Results:")
    for k, v in results[0].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
