import wandb
run = wandb.init()
artifact = run.use_artifact('social-game-rl/active-rl-planning-model/model-21d0z6dy:v14', type='model')
artifact_dir = artifact.download("./")