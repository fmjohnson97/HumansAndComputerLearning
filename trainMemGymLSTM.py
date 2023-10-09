import argparse
import os
import gymnasium as gym
import torch

from helm.trainers.lstm_trainer import LSTMPPO
# from memory_gym.mortar_mayhem_grid import GridMortarMayhemEnv
from endless_memory_gym.memory_gym.mortar_mayhem_grid import GridMortarMayhemEnv
from endless_memory_gym.memory_gym.mystery_path_grid import GridMysteryPathEnv
from endless_memory_gym.memory_gym.searing_spotlights import SearingSpotlightsEnv

def getArgs():
    parser = argparse.ArgumentParser()
    #Training Arguments
    parser.add_argument('--adv_norm', type=bool, default=False, help='')
    parser.add_argument('--clip_decay', type=str, default="none", help='')
    parser.add_argument('--clip_range', type=float, default=0.2, help='')
    parser.add_argument('--clip_range_vf', type=str, default=None, help='')
    parser.add_argument('--end_fraction', type=float, default=1, help='')
    parser.add_argument('--ent_coef', type=float, default=5e-2, help='')
    parser.add_argument('--ent_decay', type=str, default='none', help='amount of ent decay I guess')
    parser.add_argument('--ent_decay_factor', type=float, default=0.99, help='')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
    parser.add_argument('--gae_lambda', type=float, default=0.99, help='')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='')
    parser.add_argument('--min_ent_coef', type=float, default=0, help='')
    parser.add_argument('--min_lr', type=float, default=0, help='min LR')
    parser.add_argument('--n_envs', type=int, default=1, help='number of envs')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--n_steps', type=int, default=500000, help='number of steps')
    parser.add_argument('--n_rollout_steps', type=int, default=128, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial LR')
    parser.add_argument('--lr_decay', type=str, default='none', help='amount of LR decay I guess')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for Reproducibility')
    parser.add_argument('--start_fraction', type=float, default=0, help='')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='')

    #Environment Arguments
    parser.add_argument('--env', type=str, default='MM', help='the size of the memory maze environment to train on')

    #Logging Arguments
    parser.add_argument('--outpath', type=str, default='logs/', help='where to put the tensorboard logs')
    parser.add_argument('--save_ckpt', type=bool, default=True, help='to save model checkpoints or not')

    return parser.parse_args()


if __name__ == '__main__':
    config = {
        "n_batches": 8,
        "batch_size": 16,
        "beta": 100,
        "beta_lr": 1e-3,
        "beta_schedule": "none",
        "mem_len": 511,
        "min_ent_coef": 0,
        "model": "HELM",
        "optimizer": "AdamW",
        "epsilon": 1e-8,
        "topk": 1,
        "learning_rate": 1e-4}

    args = getArgs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # env = gym.make("Endless-SearingSpotlights-v0")
    # env = gym.make("Endless-MortarMayhem-v0")
    # env = gym.make("MortarMayhem-v0")
    # env = gym.make("MortarMayhemB-v0")
    # env = gym.make("Endless-MysteryPath-v0")
    # env = gym.make("MysteryPath-v0")

    if args.env == 'MM':
        # env = gym.make("MortarMayhem-Grid-v0")
        # env = gym.make("MortarMayhemB-Grid-v0")
        env = GridMortarMayhemEnv(render_mode="rgb_array")
    elif args.env == 'MP':
        env = GridMysteryPathEnv(render_mode="rgb_array")
    elif args.env == 'SS':
        env = SearingSpotlightsEnv(render_mode="rgb_array")
    else:
        print(args.env,"is not a valid environment!!!")
        breakpoint()

    model = LSTMPPO("MlpPolicy", env, verbose=1, tensorboard_log=args.outpath,lr_decay=args.lr_decay,
                        ent_coef=args.ent_coef, ent_decay=args.ent_decay, learning_rate=args.learning_rate,
                        vf_coef=args.vf_coef, n_epochs=args.n_epochs, ent_decay_factor=args.ent_decay_factor,
                        clip_range=args.clip_range, gamma=args.gamma, gae_lambda=args.gae_lambda,
                        n_steps=args.n_rollout_steps, n_envs=args.n_envs, min_lr=args.min_lr,
                        min_ent_coef=args.min_ent_coef, start_fraction=args.start_fraction,
                        end_fraction=args.end_fraction, device=device, clip_decay=args.clip_decay,
                        config=config, clip_range_vf=args.clip_range_vf, seed=args.seed,
                        max_grad_norm=args.max_grad_norm, adv_norm=args.adv_norm,
                        save_ckpt=args.save_ckpt)

    model.learn(total_timesteps=args.n_steps, eval_log_path=args.outpath)
    # iteration = 0
    # total_timesteps, callback = model._setup_learn( args.n_steps, None, None, 1, 100, args.outpath, True, "PPO")
    # latest_run = model.get_latest_run_id(model.tensorboard_log, 'PPO')
    # model.save_path = os.path.join(os.path.join(model.tensorboard_log, f'PPO_{latest_run}'))
    # callback.on_training_start(locals(), globals())
    # model._dump_sources(model.save_path)
    # model._dump_config(model.save_path)
    #
    # while model.num_timesteps < total_timesteps:
    #     continue_training = model.collect_rollouts(model.env, callback, model.rollout_buffer, model.n_steps)
    #
    #     if continue_training is False:
    #         break
    #
    #     iteration += 1
    #     model._update_current_progress_remaining(model.num_timesteps, total_timesteps)
    #     breakpoint()

    env.close()
