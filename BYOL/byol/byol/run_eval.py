print('------running eval code------')
import logging
import torch
import pytorch_lightning as pl
import wandb
print('------ imported libraries')

from paths import Path_Handler
# from vae import CNNVAE
from models import BYOL
from evaluation import Linear_Eval, Lightning_Eval
from datamodules import RGZ_DataModule
print('------ imported py files')

def run_linear_eval(config, encoder, datamodule, logger):
    print('------running linear eval')
    ## Trainer and logger ##
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=10,  # only 1 epoch since training is outside the LightningModule
        **config["trainer"],
    )

    # Create the evaluation LightningModule
    model = Lightning_Eval(config)
    model.encoder = encoder 
    print('------eval model acquired')

    # Run validation & test (evaluation is handled in the validation hooks)
    trainer.validate(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    print('------linear eval done')
    return model


def main():
    print('------in main')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    paths = Path_Handler()._dict()
    
    manual_ckpt_path = "/share/nas2_3/jalphonse/BYOL/byol/checkpoints/ajrehorg/last.ckpt"
    byolmodel = BYOL.load_from_checkpoint(manual_ckpt_path)
    
    config = byolmodel.config
    print('------byol model and config acquired')


    # Logging
    project_name = "BYOL_linear_eval"
    wandb.init(project=project_name, config=config)

    logger = pl.loggers.WandbLogger(
        project=project_name,
        save_dir=paths["files"] / "linear_eval" / str(wandb.run.id),
        reinit=True,
        config=config,
    )
    
    all_seeds = []
    all_models = []

    # Run evaluation for each seed (or just one)
    for seed in range(1):
        # Seed everything
        pl.seed_everything(seed)

        # Load data module for evaluation
        eval_datamodule = RGZ_DataModule(
        path=paths["rgz"],
        batch_size=config["data"]["batch_size"],
        center_crop=config["augmentations"]["center_crop"],
        cut_threshold=config["data"]["cut_threshold"],
        prefetch_factor=config["dataloading"]["prefetch_factor"],
        num_workers=config["dataloading"]["num_workers"],
        )
        print('------Datamodule acquired')

        model = run_linear_eval(config, byolmodel.encoder, eval_datamodule, logger)
        all_models.append(model)
        all_seeds.append(seed)


        # Log per-seed metrics
        wandb.log({f"eval/seed_{seed}_done": 1})
        print(f"------ Evaluation complete for seed {seed}")

    print("\nAll seeds evaluated:", all_seeds)

    wandb.log({"eval/seeds_completed": all_seeds})

    logger.experiment.finish()
    wandb.finish()
    print('------Evaluation complete')
    

if __name__ == "__main__":
    main()