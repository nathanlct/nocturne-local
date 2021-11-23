from pathlib import Path
import os
import time

import hydra
import imageio
import numpy as np
import setproctitle
import torch
import wandb

from algos.ppo.base_runner import Runner
from algos.ppo.env_wrappers import SubprocVecEnv, DummyVecEnv, ShareDummyVecEnv, ShareSubprocVecEnv

from nocturne_utils.wrappers import create_ppo_env

os.environ["DISPLAY"] = ":0.0"

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


def make_train_env(cfg):
    def get_env_fn(rank):
        def init_env():
            env = create_ppo_env(cfg, rank)
            # TODO(eugenevinitsky) implement this
            env.seed(cfg.seed + rank * 1000)
            return env
        return init_env
    if cfg.algo.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(cfg.algo.n_rollout_threads)])


def make_eval_env(cfg):
    def get_env_fn(rank):
        def init_env():
            env = create_ppo_env(cfg)
            # TODO(eugenevinitsky) implement this
            env.seed(cfg.seed + rank * 1000)
            return env
        return init_env
    if cfg.algo.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(cfg.algo.n_eval_rollout_threads)])

class NocturneSharedRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the Nocturne envs. Assumes a shared policy."""
    def __init__(self, config):
        super(NocturneSharedRunner, self).__init__(config)
        self.cfg = config['cfg.algo']

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads // self.episodes_per_thread

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for _ in range(self.episodes_per_thread):
                done_initialized = False
                for step in range(self.episode_length):
                    # Sample actions
                    values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                        
                    # Obser reward and next obs
                    obs, rewards, dones, infos = self.envs.step(actions_env)

                    # the episode gets reset if all agents die so we need to persist the masks across the death
                    # TODO(eugenevinitsky) remove this once it works more sensibly 
                    if done_initialized:
                        done_tracker = np.logical_or(dones, done_tracker)
                    # TODO(eugenevinitsky) this assumes that the first time-step is never done
                    if not done_initialized:
                        done_initialized = True
                        done_tracker = np.zeros_like(dones)

                    # if np.all(dones):
                    #     import ipdb; ipdb.set_trace()

                    data = obs, rewards, done_tracker, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                    # insert data into buffer
                    self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads * self.episodes_per_thread
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.algorithm_name,
                                self.experiment_name,
                                episode * self.episodes_per_thread,
                                episodes * self.episodes_per_thread,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                env_infos = {}
                for agent_id in range(self.num_agents):
                    idv_rews = []
                    for info in infos:
                        if 'individual_reward' in info[agent_id].keys():
                            idv_rews.append(info[agent_id]['individual_reward'])
                    agent_k = 'agent%i/individual_rewards' % agent_id
                    env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards * self.buffer.masks[:-1]) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                print(f"maximum per step reward is {np.max(self.buffer.rewards)}")
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

            # save videos 
            if episode % self.cfg.render_interval == 0:
                self.render(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []

            if eval_episode >= self.cfg.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                break

    @torch.no_grad()
    def render(self, total_num_steps):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.cfg.render_episodes):
            done_initialized = False
            obs = envs.reset()
            if self.cfg.save_gifs:
                image = envs.render('rgb_array')[0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            self.trainer.prep_rollout()
            for step in range(self.episode_length):
                calc_start = time.time()

                # TODO(eugenevinitsky) put back deterministic = True
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=False)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                if done_initialized:
                        done_tracker = np.logical_or(dones, done_tracker)
                # TODO(eugenevinitsky) this assumes that the first time-step is never done
                if not done_initialized:
                    done_initialized = True
                    done_tracker = np.zeros_like(dones)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.cfg.save_gifs:
                    image = envs.render('rgb_array')[0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.cfg.ifi:
                        time.sleep(self.cfg.ifi - elapsed)
                else:
                    envs.render('human')
                
                print(dones[0])
                if np.all(dones[0]):
                    print('exited rendering due to episode termination')
                    break

            print("episode reward of rendered episode is: " + str(np.mean(np.sum(np.array(episode_rewards)[:, 0], axis=0))))

        if self.cfg.save_gifs:
            # if self.use_wandb:
            #     np_arr = np.stack(all_frames).transpose((0, 3, 1, 2))
            #     wandb.log({"video": wandb.Video(np_arr, fps=4, format="gif")}, step=total_num_steps)
            # else:
            imageio.mimsave(os.getcwd() + '/render.gif', all_frames, duration=self.cfg.ifi)

@hydra.main(config_path='../../cfgs/', config_name='config')
def main(cfg):
    logdir = Path(os.getcwd())
    if cfg.wandb_id is not None:
        wandb_id = cfg.wandb_id
    else:
        wandb_id = wandb.util.generate_id()
        # with open(os.path.join(logdir, 'wandb_id.txt'), 'w+') as f:
        #     f.write(wandb_id)
    wandb_mode = "disabled" if (cfg.debug or not cfg.wandb) else "online"

    if cfg.wandb:
        run = wandb.init(config=cfg,
                        project=cfg.wandb_name,
                        name=wandb_id,
                        group='ppov2_' + cfg.experiment,
                        resume="allow",
                        settings=wandb.Settings(start_method="fork"),
                        mode=wandb_mode)
    else:
        if not logdir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in logdir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        logdir = logdir / curr_run
        if not logdir.exists():
            os.makedirs(str(logdir))

    if cfg.algo.algorithm_name == "rmappo":
        assert (cfg.algo.use_recurrent_policy or cfg.algo.use_naive_recurrent_policy), ("check recurrent policy!")
    elif cfg.algo.algorithm_name == "mappo":
        assert (cfg.algo.use_recurrent_policy == False and cfg.algo.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if 'cpu' not in cfg.algo.device and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(cfg.algo.device)
        torch.set_num_threads(cfg.algo.n_training_threads)
        if cfg.algo.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(cfg.algo.n_training_threads)

    setproctitle.setproctitle(str(cfg.algo.algorithm_name) + "-" + str(cfg.experiment))

    # seed
    torch.manual_seed(cfg.algo.seed)
    torch.cuda.manual_seed_all(cfg.algo.seed)
    np.random.seed(cfg.algo.seed)

    # env init
    envs = make_train_env(cfg)
    eval_envs = make_eval_env(cfg) if cfg.algo.use_eval else None
    # TODO(eugenevinitsky) hacky
    num_agents = envs.reset().shape[1]

    config = {
        "cfg.algo": cfg.algo,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "logdir": logdir
    }

    # run experiments
    runner = NocturneSharedRunner(config)
    runner.run()
    
    # post process
    envs.close()
    if cfg.algo.use_eval and eval_envs is not envs:
        eval_envs.close()

    if cfg.wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

if __name__ == '__main__':
    main()