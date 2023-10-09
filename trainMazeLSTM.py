import argparse

import gym
import torch

from helm.trainers.lstm_trainer import LSTMPPO
# from memory_maze.memory_maze.gym_wrappers import GymWrapper


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
    parser.add_argument('--n_envs', type=int, default=16, help='number of envs')
    parser.add_argument('--n_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--n_steps', type=int, default=500000, help='number of steps')
    parser.add_argument('--n_rollout_steps', type=int, default=128, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial LR')
    parser.add_argument('--lr_decay', type=str, default='none', help='amount of LR decay I guess')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for Reproducibility')
    parser.add_argument('--start_fraction', type=float, default=0, help='')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='')

    #Environment Arguments
    parser.add_argument('--env', type=str, default='9x9', help='the size of the memory maze environment to train on')

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
        "topk": 1 }

    args = getArgs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.env == '9x9':
        env = gym.make('memory_maze:MemoryMaze-9x9-v0')
    elif args.env == '11x11':
        env = gym.make('memory_maze:MemoryMaze-11x11-v0')
    elif args.env == '13x13':
        env = gym.make('memory_maze:MemoryMaze-13x13-v0')
    elif args.env == '15x15':
        env = gym.make('memory_maze:MemoryMaze-15x15-v0')
    else:
        print(args.env,"is not a valid environment size!!!")
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

    env.close()
