# Agent Environments - SearchQA

## Setup

```sh
# Option A: uv (recommended for lightweight local dev)
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv not installed
uv venv .uv
uv pip install -e ../agentenv      # install core agentenv (adjust path if needed)
uv pip install -e .                # install this SearchQA environment
uv pip install -r requirements.txt # if a requirements.txt exists
bash ./setup.sh                    # downloads / builds data artifacts

# Option B: conda (legacy)
conda env create -f environment.yml
conda activate agentenv-searchqa
pip install -e .
bash ./setup.sh
```

## Launch

```sh
# Using installed console script (uv environment already created)
uv run searchqa --host 0.0.0.0 --port 36001

# Direct module invocation
uv run python -m agentenv_searchqa.launch --host 0.0.0.0 --port 36001
```

## Environment variables

`SEARCHQA_FAISS_GPU`: Force enable RAG server on GPUs

> Other variables please refer to `env_warpper.py` line 50-68

## Makefile Integration

The repository root `Makefile` provides convenience targets:

```sh
make agentenv-searchqa-setup   # create uv venv & install
make agentenv-searchqa-server  # start server
make agentenv-searchqa-eval    # run eval (server must be up)
make agentenv-searchqa-clean   # remove venv/logs
```

Adjust ports and paths via environment variables, e.g.:

```sh
SEARCHQA_PORT=36005 make agentenv-searchqa-server
```


## Item ID

| Item ID         | Description             | Split |
| --------------- | ----------------------- | ----- |
| 0 ~ 3609        | nq Dataset              | Test  |
| 3610 ~ 14922    | triviaqa Dataset        | Test  |
| 14923 ~ 29189   | popqa Dataset           | Test  |
| 29190 ~ 36594   | hotpotqa Dataset        | Test  |
| 36595 ~ 49170   | 2wikimultihopqa Dataset | Test  |
| 49171 ~ 51587   | musique Dataset         | Test  |
| 51588 ~ 51712   | bamboogle Dataset       | Test  |
| 51713 ~ 130880  | nq Dataset              | Train |
| 130881 ~ 221328 | hotpotqa Dataset        | Train |
