import wandb
run = wandb.init()
artifact = run.use_artifact('social-game-rl/active-rl-planning-model/model-3u5r2w4i:v26', type='model')
artifact_dir = artifact.download("./models")