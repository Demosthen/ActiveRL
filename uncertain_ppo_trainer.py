import warnings
from typing import Any, Callable, Dict, Optional, Type, Union, List

import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces
from torch.nn import functional as F

from ray.rllib.algorithms.ppo import PPO

import ray
import logging
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.evaluation.metrics import (
    collect_episodes,
    collect_metrics,
    summarize_episodes,
)
from ray.exceptions import GetTimeoutError, RayActorError, RayError
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch, concat_samples
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_SAMPLED_THIS_ITER,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_THIS_ITER,
    NUM_ENV_STEPS_TRAINED,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TRAINING_ITERATION_TIMER,
)


from uncertain_ppo import UncertainPPOTorchPolicy

logger = logging.getLogger(__name__)

class UncertainPPO(PPO):

    @override(PPO)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":

            return UncertainPPOTorchPolicy
        else:
            return NotImplementedError


    @override(PPO)
    def evaluate(
        self,
        duration_fn: Optional[Callable[[int], int]] = None,
    ) -> dict:
        """Evaluates current policy under `evaluation_config` settings.
        Note that this default implementation does not do anything beyond
        merging evaluation_config with the normal trainer config.
        Args:
            duration_fn: An optional callable taking the already run
                num episodes as only arg and returning the number of
                episodes left to run. It's used to find out whether
                evaluation should continue.
        """
        # Call the `_before_evaluate` hook.
        self._before_evaluate()

        # Sync weights to the evaluation WorkerSet.
        if self.evaluation_workers is not None:
            self.evaluation_workers.sync_weights(
                from_worker=self.workers.local_worker()
            )
            self._sync_filters_if_needed(
                from_worker=self.workers.local_worker(),
                workers=self.evaluation_workers,
                timeout_seconds=self.config[
                    "sync_filters_on_rollout_workers_timeout_s"
                ],
            )

        self.callbacks.on_evaluate_start(algorithm=self)

        if self.config["custom_eval_function"]:
            logger.info(
                "Running custom eval function {}".format(
                    self.config["custom_eval_function"]
                )
            )
            metrics = self.config["custom_eval_function"](self, self.evaluation_workers)
            if not metrics or not isinstance(metrics, dict):
                raise ValueError(
                    "Custom eval function must return "
                    "dict of metrics, got {}.".format(metrics)
                )
        else:
            if (
                self.evaluation_workers is None
                and self.workers.local_worker().input_reader is None
            ):
                raise ValueError(
                    "Cannot evaluate w/o an evaluation worker set in "
                    "the Trainer or w/o an env on the local worker!\n"
                    "Try one of the following:\n1) Set "
                    "`evaluation_interval` >= 0 to force creating a "
                    "separate evaluation worker set.\n2) Set "
                    "`create_env_on_driver=True` to force the local "
                    "(non-eval) worker to have an environment to "
                    "evaluate on."
                )

            # How many episodes/timesteps do we need to run?
            # In "auto" mode (only for parallel eval + training): Run as long
            # as training lasts.
            unit = self.config["evaluation_duration_unit"]
            eval_cfg = self.config["evaluation_config"]
            rollout = eval_cfg["rollout_fragment_length"]
            num_envs = eval_cfg["num_envs_per_worker"]
            auto = self.config["evaluation_duration"] == "auto"
            duration = (
                self.config["evaluation_duration"]
                if not auto
                else (self.config["evaluation_num_workers"] or 1)
                * (1 if unit == "episodes" else rollout)
            )
            agent_steps_this_iter = 0
            env_steps_this_iter = 0

            # Default done-function returns True, whenever num episodes
            # have been completed.
            if duration_fn is None:

                def duration_fn(num_units_done):
                    return duration - num_units_done

            logger.info(f"Evaluating current policy for {duration} {unit}.")

            metrics = None
            all_batches = []
            # No evaluation worker set ->
            # Do evaluation using the local worker. Expect error due to the
            # local worker not having an env.
            if self.evaluation_workers is None:
                # If unit=episodes -> Run n times `sample()` (each sample
                # produces exactly 1 episode).
                # If unit=ts -> Run 1 `sample()` b/c the
                # `rollout_fragment_length` is exactly the desired ts.
                iters = duration if unit == "episodes" else 1
                for _ in range(iters):
                    batch = self.workers.local_worker().sample()
                    agent_steps_this_iter += batch.agent_steps()
                    env_steps_this_iter += batch.env_steps()
                    if self.reward_estimators:
                        all_batches.append(batch)
                metrics = collect_metrics(
                    self.workers.local_worker(),
                    keep_custom_metrics=eval_cfg["keep_per_episode_custom_metrics"],
                    timeout_seconds=eval_cfg["metrics_episode_collection_timeout_s"],
                )

            # Evaluation worker set only has local worker.
            elif self.config["evaluation_num_workers"] == 0:
                # If unit=episodes -> Run n times `sample()` (each sample
                # produces exactly 1 episode).
                # If unit=ts -> Run 1 `sample()` b/c the
                # `rollout_fragment_length` is exactly the desired ts.
                iters = duration if unit == "episodes" else 1
                for _ in range(iters):
                    batch = self.evaluation_workers.local_worker().sample()
                    agent_steps_this_iter += batch.agent_steps()
                    env_steps_this_iter += batch.env_steps()
                    if self.reward_estimators:
                        all_batches.append(batch)

            # Evaluation worker set has n remote workers.
            else:
                # How many episodes have we run (across all eval workers)?
                num_units_done = 0
                _round = 0
                while True:
                    units_left_to_do = duration_fn(num_units_done)
                    if units_left_to_do <= 0:
                        break

                    _round += 1
                    try:
                        batches = ray.get(
                            [
                                w.sample.remote()
                                for i, w in enumerate(
                                    self.evaluation_workers.remote_workers()
                                )
                                if i * (1 if unit == "episodes" else rollout * num_envs)
                                < units_left_to_do
                            ],
                            timeout=self.config["evaluation_sample_timeout_s"],
                        )
                    except GetTimeoutError:
                        logger.warning(
                            "Calling `sample()` on your remote evaluation worker(s) "
                            "resulted in a timeout (after the configured "
                            f"{self.config['evaluation_sample_timeout_s']} seconds)! "
                            "Try to set `evaluation_sample_timeout_s` in your config"
                            " to a larger value."
                            + (
                                " If your episodes don't terminate easily, you may "
                                "also want to set `evaluation_duration_unit` to "
                                "'timesteps' (instead of 'episodes')."
                                if unit == "episodes"
                                else ""
                            )
                        )
                        break

                    _agent_steps = sum(b.agent_steps() for b in batches)
                    _env_steps = sum(b.env_steps() for b in batches)
                    # 1 episode per returned batch.
                    if unit == "episodes":
                        num_units_done += len(batches)
                        # Make sure all batches are exactly one episode.
                        for ma_batch in batches:
                            ma_batch = ma_batch.as_multi_agent()
                            for batch in ma_batch.policy_batches.values():
                                assert np.sum(batch[SampleBatch.DONES])
                    # n timesteps per returned batch.
                    else:
                        num_units_done += (
                            _agent_steps if self._by_agent_steps else _env_steps
                        )
                    if self.reward_estimators:
                        all_batches.extend(batches)

                    agent_steps_this_iter += _agent_steps
                    env_steps_this_iter += _env_steps

                    logger.info(
                        f"Ran round {_round} of parallel evaluation "
                        f"({num_units_done}/{duration if not auto else '?'} "
                        f"{unit} done)"
                    )

            if metrics is None:
                metrics = collect_metrics(
                    self.evaluation_workers.local_worker(),
                    self.evaluation_workers.remote_workers(),
                    keep_custom_metrics=self.config["keep_per_episode_custom_metrics"],
                    timeout_seconds=eval_cfg["metrics_episode_collection_timeout_s"],
                )
            metrics[NUM_AGENT_STEPS_SAMPLED_THIS_ITER] = agent_steps_this_iter
            metrics[NUM_ENV_STEPS_SAMPLED_THIS_ITER] = env_steps_this_iter
            # TODO: Remove this key at some point. Here for backward compatibility.
            metrics["timesteps_this_iter"] = env_steps_this_iter

            if self.reward_estimators:
                # Compute off-policy estimates
                metrics["off_policy_estimator"] = {}
                total_batch = concat_samples(all_batches)
                for name, estimator in self.reward_estimators.items():
                    estimates = estimator.estimate(total_batch)
                    metrics["off_policy_estimator"][name] = estimates

        # Evaluation does not run for every step.
        # Save evaluation metrics on trainer, so it can be attached to
        # subsequent step results as latest evaluation result.
        self.evaluation_metrics = {"evaluation": metrics}

        self.callbacks.on_evaluate_end(
            algorithm=self, evaluation_metrics=self.evaluation_metrics
        )

        # Also return the results here for convenience.
        return self.evaluation_metrics