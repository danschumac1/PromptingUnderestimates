# Classification_v2

This repo was designed to run all scripts at the parent directory (don't cd into folders to run scripts)

### Prerequisites
- Python 3.8+
- PyTorch 2.9.1
- CUDA-compatible GPU (we used CUDA Version: 12.3)


1. Install dependencies:
```bash
pip install -r resources/requirements.txt
```

2. Install additional packages:
```bash
pip install git+https://github.com/moment-timeseries-foundation-model/moment.git
```

3. create a `.env `file in `./resources`. Set the following variables
- HF_TOKEN
- SLACK_WEBHOOK_URL
- TOGETHER_API_KEY
- OPENAI_API_KEY
- MISTRAL_RANDOM_PROCESSOR_PATH
- MISTRAL_SMALL_31_PATH

## Datasets
The framework supports several time series classification datasets:

- **HAR**
- **HAD**
- **RWC**
- **TEE**
- **ECG** 
- **CTU**

To clean the data see `./bin/data_builders`
- `get_and_clean_data.sh`
- `USC_HAD_data.sh`

### Language Models
- **Qwen-VL**
- **Llama**
- **Mistral**
    - Mistral (and its randomly intialized version) must be downloaded ahead of time. 
    - run `python ./src/_download_mistral_random.py` and `python ./src/_download_mistral_random.py`


## Usage
### Prompting
- Classification `./bin/generation/prompting_1.sh`

### Probing
Proping happens over two stages
1. generating embeddings
    - see bin/features/embedding_all_layers_gpu0.sh
2. logistic regression on those embeddings
    - see bin/generation/logistic_regression_1.sh


### Evaluation
Evaluation scripts can be found in `./bin/eval`
run `bin/eval/logistic_regression_eval_1.sh` for probing eval
run `bin/eval/prompting_eval.sh` for prompting eval`

## Project Structure
```
├── bin/                    # Shell scripts for automation
│   ├── data_builders/      # Data preparation scripts
│   ├── eval/               # Evaluation pipelines
│   ├── features/           # Feature extraction scripts
│   └── generation/         # Model training scripts
├── data/                   # Datasets and results
├── logs/                   # Log files
├── resources/              # Configuration files
│   ├── requirements.txt    # Python dependencies
│   ├── ts_backbone.yaml    # Model configurations
│   └── .env                # Place to set your enviromental variables
└── src/                    # Python source code
    ├── data_management/    # Data loading utilities
    ├── utils/              # Helper functions
    └── visualization/      # Plotting and analysis
```