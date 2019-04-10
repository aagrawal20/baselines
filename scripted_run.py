#!venv/bin/python
import signal
import subprocess
import argparse
import os

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('env', type=str, choices=['reacher_actuation','cheetah_dithering','reacher_revisit', 'cheetah_actuation', 'breakout_dithering'])
parser.add_argument('reward_mod', type=int)
parser.add_argument('reward_mod_str', type=str)
parser.add_argument('base_dir_num', type=int)
args = parser.parse_args()


def signal_handler(signal, frame):
    global interrupted
    interrupted = True


signal.signal(signal.SIGINT, signal_handler)

interrupted = False
i = args.base_dir_num
while True:
    print("Starting another run!")
    my_env = os.environ.copy()
    if args.env == 'reacher_actuation':
        my_env["OPENAI_LOGDIR"] = "reacher_actuation_{}_{}".format(args.reward_mod_str, i)
        subprocess.call(
            'python -m baselines.run --constraints reacher_actuation_counting --rewards {}'
            .format(args.reward_mod).split(), env=my_env)
    elif args.env == 'cheetah_dithering':
        my_env["OPENAI_LOGDIR"] = "cheetah_dithering_{}_{}".format(args.reward_mod_str, i)
        subprocess.call(
            'python -m baselines.run --env HalfCheetah-v2 --constraints half_cheetah_dithering_0 half_cheetah_dithering_1 half_cheetah_dithering_2 half_cheetah_dithering_3 half_cheetah_dithering_4 half_cheetah_dithering_5 --rewards {r} {r} {r} {r} {r} {r}'
            .format(r=args.reward_mod).split(), env=my_env)
    elif args.env == 'cheetah_dithering':
        my_env["OPENAI_LOGDIR"] = "cheetah_actuation_{}_{}".format(args.reward_mod_str, i)
        subprocess.call(
            'python -m baselines.run --env HalfCheetah-v2 --constraints half_cheetah_actuation_0 half_cheetah_actuation_1 half_cheetah_actuation_2 half_cheetah_actuation_3 half_cheetah_actuation_4 half_cheetah_actuation_5 --rewards {r} {r} {r} {r} {r} {r}'
            .format(r=args.reward_mod).split(), env=my_env)
    elif args.env == 'reacher_revisit':
        my_env["OPENAI_LOGDIR"] = "reacher_revisit_{}_{}".format(args.reward_mod_str, i)
        subprocess.call(
            'python -m baselines.run --constraints reacher_revisit_counting --rewards {}'
            .format(args.reward_mod).split(), env=my_env)
    elif args.env == 'breakout_dithering':
        my_env["OPENAI_LOGDIR"] = "breakout_dithering_{}_{}".format(args.reward_mod_str, i)
        subprocess.call(
            'python -m baselines.run --env BreakoutNoFrameskip-v4 --num_timesteps 1e7 --alg deepq --constraints 1d_dithering --rewards {}'
            .format(args.reward_mod).split(), env=my_env)


    print("All done!")
    i += 1

    if interrupted:
        print("Gotta go")
        break
