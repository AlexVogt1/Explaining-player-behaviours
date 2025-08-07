import os
import wandb

run =wandb.init(project="gym-md_baselines", job_type="add heat")
artifact = wandb.Artifact(name="heatmaps_1", type="heatmap")
artifact.add_dir(local_path='./heatmaps')
run.log_artifact(artifact)
