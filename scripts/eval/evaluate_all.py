HOMEDIR = '/home1/09720/yhpark/rl/multi-turn-rl'
WORKDIR = '/work/09720/yhpark'

import sys
sys.path.append(HOMEDIR)

import argparse
import os
from scripts.eval.eval import main
from tqdm import tqdm

buyers = ['fairness', 'length', 'supervised', 'utility']
sellers = ['brac', 'cql', 'emaq']

if __name__ == "__main__":
    args = argparse.Namespace()
    args.data_path = os.path.join(HOMEDIR, 'data/test.json')
    args.gpt_dir = os.path.join(WORKDIR, 'logs/gpt2/checkpoint-6000')
    args.seller = 'ours'
    args.num_outputs = 5
    args.debug = False
    args.num_rollouts = 50
    args.scenario = None
    
    for buyer in tqdm(buyers, position=0):
        for seller in tqdm(sellers, position=1, leave=False):
            args.output_path = os.path.join(WORKDIR, f'results_test/{buyer}_vs_{seller}.json')
            args.checkpoint_file = os.path.join(WORKDIR, f'logs/chai_{seller}/snapshot_4999.pkl')
            args.buyer = f'theirs_{buyer}'

            main(args)
