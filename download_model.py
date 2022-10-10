import wandb
run = wandb.init()
artifact = run.use_artifact('social-game-rl/active-rl-planning-model/model-qmd8tx5n:v10', type='model')
artifact_dir = artifact.download("./models")