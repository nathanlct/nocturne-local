import hydra
from omegaconf import OmegaConf
from pyvirtualdisplay import Display
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from nocturne_utils.wrappers import create_env


class RLlibWrapperEnv(MultiAgentEnv):
    metadata = {
        "render.modes": ["rgb_array"],
    }

    def __init__(self, env):
        self._skip_env_checking = True  # temporary fix for rllib env checking issue
        super().__init__()
        self._env = env

    def step(self, actions):
        next_obs, rew, done, info = self._env.step(actions)
        return next_obs, rew, done, info

    def reset(self):
        obses = self._env.reset()
        return obses

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def render(self, mode=None):
        return self._env.render()

    def seed(self, seed=None):
        self._env.seed(seed)

    def __getattr__(self, name):
        return getattr(self._env, name)


def create_rllib_env(cfg):
    return RLlibWrapperEnv(create_env(cfg))


@hydra.main(config_path="../../cfgs/", config_name="config")
def main(cfg):
    disp = Display()
    disp.start()
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # TODO(eugenevinitsky) move these into a config
    if cfg['debug']:
        ray.init(local_mode=True)
        num_workers = 0
        num_envs_per_worker = 1
        num_gpus = 0
        use_lstm = False
    else:
        num_workers = 15
        num_envs_per_worker = 5
        num_gpus = 1
        use_lstm = True

    register_env("nocturne", lambda cfg: create_rllib_env(cfg))

    tune.run(
        "PPO",
        # TODO(eugenevinitsky) move into config
        local_dir="/checkpoint/eugenevinitsky/nocturne/ray_results",
        stop={"episodes_total": 60000},
        checkpoint_freq=1000,
        config={
            # Enviroment specific.
            "env":
            "nocturne",
            "env_config":
            cfg,
            # General
            "framework":
            "torch",
            "num_gpus":
            num_gpus,
            "num_workers":
            num_workers,
            "num_envs_per_worker":
            num_envs_per_worker,
            "observation_filter":
            "MeanStdFilter",
            # Method specific.
            "entropy_coeff":
            0.0,
            "num_sgd_iter":
            5,
            "train_batch_size":
            max(100 * num_workers * num_envs_per_worker, 512),
            "rollout_fragment_length":
            20,
            "sgd_minibatch_size":
            max(int(100 * num_workers * num_envs_per_worker / 4), 512),
            "multiagent": {
                # We only have one policy (calling it "shared").
                # Class, obs/act-spaces, and config will be derived
                # automatically.
                "policies": {"shared_policy"},
                # Always use "shared" policy.
                "policy_mapping_fn":
                (lambda agent_id, episode, **kwargs: "shared_policy"),
                # each agent step is counted towards train_batch_size
                # rather than environment steps
                "count_steps_by":
                "agent_steps",
            },
            "model": {
                "use_lstm": use_lstm
            },
            # Evaluation stuff
            "evaluation_interval":
            50,
            # Run evaluation on (at least) one episodes
            "evaluation_duration":
            1,
            # ... using one evaluation worker (setting this to 0 will cause
            # evaluation to run on the local evaluation worker, blocking
            # training until evaluation is done).
            # TODO: if this is not 0, it seems to error out
            "evaluation_num_workers":
            0,
            # Special evaluation config. Keys specified here will override
            # the same keys in the main config, but only for evaluation.
            "evaluation_config": {
                # Store videos in this relative directory here inside
                # the default output dir (~/ray_results/...).
                # Alternatively, you can specify an absolute path.
                # Set to True for using the default output dir (~/ray_results/...).
                # Set to False for not recording anything.
                "record_env": "videos_test",
                # "record_env": "/Users/xyz/my_videos/",
                # Render the env while evaluating.
                # Note that this will always only render the 1st RolloutWorker's
                # env and only the 1st sub-env in a vectorized env.
                "render_env": True,
            },
        },
    )


if __name__ == "__main__":
    main()
