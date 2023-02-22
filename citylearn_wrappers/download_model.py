import wandb
run = wandb.init()
artifact = run.use_artifact('social-game-rl/active-rl-planning-model/model-1qs0ys0o:v26', type='model')
artifact_dir = artifact.download("./models")
