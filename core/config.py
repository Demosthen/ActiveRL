class ExperimentConfig():
    
    def __init__(self, env_fn, callback_fn, full_eval_fn=None, env_config={}, rllib_config={}, model_config={}, eval_env_config={}) -> None:
        """Configuration parameters for an experiment.

        Args:
        env_fn: function to call that instantiates the desired Gym env
        callback_fn: function to call to instantiate the desired RLLib Callback
        rllib_config: dictionary of parameters to override in the RLLib config
        model_config: dictionary of parameters to override in the RLLib model config
        full_eval_fn: an optional function that specifies when to stop a full evaluation.
                        See the argument to the RLLib evaluate function for details.
        env_config: dictionary of parameters to override in the RLLib env config when training.
        eval_env_config: dictionary of parameters to override in the RLLib env config when evaluating.
        """
        self.env_fn = env_fn
        self.callback_fn = callback_fn
        self.rllib_config = rllib_config
        self.model_config = model_config
        self.full_eval_fn = full_eval_fn
        self.env_config = env_config
        self.eval_env_config = eval_env_config