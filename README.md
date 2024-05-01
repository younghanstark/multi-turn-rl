# Multi-turn Reinforcement Learning with Language Models

UT Austin C S 394R (Spring 2024) Final Project

Younghan Park, Krystal An

Here's the [link](https://www.youtube.com/watch?v=xPCUbzo2v6w) to our quick demo video.

## Install Codebase & Dependencies
Start by cloning this repository. The minimum python version required for this repo is `3.8`. Install required packages by running the following command:

```bash
conda create -n chai python=3.8
conda activate chai
pip install -r requirements.txt
```

### Note to TACC Users
All the preprocessing steps and experiments are done in a single NVIDIA A100 node. Therefore, all the workloads should work in the partition `gpu-a100-small` in lonestar6. You can form a slurm command that works in lonestar6 with the following template.

```bash
sbatch -J {job name} -o {output file} -e {error file} -p gpu-a100-small -N 1 -t {estimated required time e.g., 4:00:00} --wrap "{the command you want to run with GPU}"
```

## Finetuning the Language Model
We use GPT-2 as our basic language model. To finetune GPT-2 on the CraigslistBargain dataset, you can run the following command in `scripts/transformers`:

``` bash
  python finetune_gpt2.py \
	 --gpt2-type gpt2-medium \
	 --train-fp ../../data/train.json \
	 --val-fp ../../data/dev.json \
	 --batch-size 16 \
	 --output-dir <log_dir>
```

Note that `16` is the largest batch size that we can use in single TACC node.

We then use the Finetuned GPT to generate candidates and embeddings via the following command. To set the output path, see the source code and use appropriate command line argument:

``` bash
  python generate_sentences.py <checkpoint_dir>
  python generate_embeddings.py <checkpoint_dir>
```

where `checkpoint_dir` is the directory created from the previous command, for example, `<log_dir>/checkpoint-6000/`. This should create `sentences.pkl` and `embeddings.pkl` files.

## Training the RL agent
After generating the candidate sentences and embeddings, we can train CHAI agents by using the scripts in the `scripts/train` directory. For example, to train CHAI agent with EMaQ algorithm, run:

``` bash
  python chai_emaq.py \
	   --logdir <log_dir> \
	   --filepath ./data/train.json \
	   --embeddings <path_to_embeddings.pkl> \
	   --sentences <path_to_sentences.pkl>
```

You can use different hyperparameter such as batch size, or epochs in the command line. For more information, you can refer to the training script.

## Evaluate the model
To try some negotiation against the trained agent, run following:

``` bash
  python eval.py \
	 --data-path ../../test.json \
	 --gpt-dir <gpt_checkpoint> \
	 --checkpoint-file <rl_agent_checkpoint> \
	 --buyer human \
	 --seller ours \
	 --num-rollouts 50 \
	 --output-path <path_to_output_json> \
	 --debug
```

The `eval.py` script also contains options for the automatic evaluations done in the paper, which can be done by changing the `buyer` argument to different values such as `theirs_utility`, or `theirs_supervised`. Check out the script for more details.
