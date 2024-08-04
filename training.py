import re
import torch
import shutil
import os
from pathlib import Path
from datasets import load_dataset, DatasetDict, Translation
from joeynmt.config import load_config, parse_global_args
from joeynmt.prediction import predict, prepare
from huggingface_hub import snapshot_download, HfApi

def remove_special_characters(text, chars_to_remove_regex):
    text = re.sub(chars_to_remove_regex, '', text)
    text = text.lower()
    return text.strip()

def clean_text(batch, src_lang, trg_lang, chars_to_remove_regex):
    batch['translation'][src_lang] = remove_special_characters(batch['translation'][src_lang], chars_to_remove_regex)
    batch['translation'][trg_lang] = remove_special_characters(batch['translation'][trg_lang], chars_to_remove_regex)
    return batch

def create_config(data_dir, model_dir):
    config = f"""
name: "dyu_fr_transformer-sp"
joeynmt_version: "2.3.0"
model_dir: "{model_dir}"
use_cuda: True # False for CPU training
fp16: True

data:
    train: "{data_dir}"
    dev: "{data_dir}"
    test: "{data_dir}"
    dataset_type: "huggingface"
    dataset_cfg:
        name: "dyu-fr"
    sample_dev_subset: 1460
    src:
        lang: "dyu"
        max_length: 100
        lowercase: False
        normalize: False
        level: "bpe"
        voc_limit: 2000
        voc_min_freq: 1
        voc_file: "{data_dir}/vocab.txt"
        tokenizer_type: "sentencepiece"
        tokenizer_cfg:
            model_file: "{data_dir}/sp.model"
    trg:
        lang: "fr"
        max_length: 100
        lowercase: False
        normalize: False
        level: "bpe"
        voc_limit: 2000
        voc_min_freq: 1
        voc_file: "{data_dir}/vocab.txt"
        tokenizer_type: "sentencepiece"
        tokenizer_cfg:
            model_file: "{data_dir}/sp.model"
    special_symbols:
        unk_token: "<unk>"
        unk_id: 0
        pad_token: "<pad>"
        pad_id: 1
        bos_token: "<s>"
        bos_id: 2
        eos_token: "</s>"
        eos_id: 3
"""
    return config

def append_training_testing_config(config, data_dir, model_dir):
    additional_config = f"""
testing:
    #load_model: "{model_dir}/best.ckpt"
    n_best: 1
    beam_size: 5
    beam_alpha: 1.0
    batch_size: 1024
    batch_type: "token"
    max_output_length: 100
    eval_metrics: ["bleu"]
    sacrebleu_cfg:
        tokenize: "13a"

training:
    #load_model: "{model_dir}/latest.ckpt"
    #reset_best_ckpt: False
    #reset_scheduler: False
    #reset_optimizer: False
    #reset_iter_state: False
    random_seed: 42
    optimizer: "adamw"
    normalization: "tokens"
    adam_betas: [0.9, 0.98]
    scheduling: "plateau"
    learning_rate_warmup: 100
    learning_rate: 0.0003
    learning_rate_min: 0.00000001
    weight_decay: 0.0
    label_smoothing: 0.1
    loss: "crossentropy"
    batch_size: 128
    batch_type: "token"
    batch_multiplier: 4
    early_stopping_metric: "bleu"
    epochs: 208
    validation_freq: 2000
    logging_freq: 2000
    overwrite: True
    shuffle: True
    print_valid_sents: [0, 1, 2, 3]
    keep_best_ckpts: 3

model:
    initializer: "xavier_uniform"
    bias_initializer: "zeros"
    init_gain: 1.0
    embed_initializer: "xavier_uniform"
    embed_init_gain: 1.0
    tied_embeddings: True
    tied_softmax: True
    encoder:
        type: "transformer"
        num_layers: 4
        num_heads: 8
        embeddings:
            embedding_dim: 256
            scale: True
            dropout: 0.1
        hidden_size: 256
        ff_size: 1024
        dropout: 0.1
        layer_norm: "pre"
        activation: "gelu"
    decoder:
        type: "transformer"
        num_layers: 4
        num_heads: 8
        embeddings:
            embedding_dim: 256
            scale: True
            dropout: 0.1
        hidden_size: 256
        ff_size: 1024
        dropout: 0.1
        layer_norm: "pre"
        activation: "gelu"
"""
    config += additional_config
    return config

def save_and_train(config, data_dir):
    with (Path(data_dir) / "config.yaml").open('w') as f:
        f.write(config)
    
    os.system(f"python build_vocab.py {data_dir}/config.yaml --joint")
    os.system(f"python -m joeynmt train {data_dir}/config.yaml --skip-test")

def update_config_with_model_info(model_dir, data_dir):
    with (Path(model_dir) / "config.yaml").open('r') as f:
        config = f.read()
    
    updated_config = config\
        .replace(f'#load_model: "{model_dir}/best.ckpt"', f'load_model: "{model_dir}/best.ckpt"')\
        .replace(f'model_file: "{data_dir}/sp.model"', f'model_file: "{model_dir}/sp.model"')\
        .replace(f'voc_file: "{data_dir}/vocab.txt"', f'voc_file: "{model_dir}/vocab.txt"')
    
    with (Path(model_dir) / "config.yaml").open('w') as f:
        f.write(updated_config)

def copy_files(src_files, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for src, dst in src_files:
        shutil.copy(src, dst)

def create_model_card(HF_REPO_NAME, lean_model_dir):
    model_card = f"""---
    language:
    - en
    - fr
    - multilingual
    tags:
    - translation
    - pytorch
    model-index:
    - name: koleshjr/dyu-fr-joeynmt
    results: []
    ---

    # koleshjr/dyu-fr-joeynmt

    An example of a machine translation model that translates Dyula to French using the [JoeyNMT framework](https://github.com/joeynmt/joeynmt).

    This following example is based on [this Github repo](https://github.com/data354/koumakanMT-challenge) that was kindly created by [data354](https://data354.com/en/).

    ## Model description

    More information needed

    ## Intended uses & limitations

    More information needed

    ## Training and evaluation data

    More information needed

    ## Usage

    ### Load and use for inference

    ```python
    import torch
    from joeynmt.config import load_config, parse_global_args
    from joeynmt.prediction import predict, prepare
    from huggingface_hub import snapshot_download

    # Download model
    snapshot_download(
        repo_id="{HF_REPO_NAME}",
        local_dir="/path/to/save/locally"
    )

    # Define model interface
    class JoeyNMTModel:
        '''
        JoeyNMTModel which load JoeyNMT model for inference.

        :param config_path: Path to YAML config file
        :param n_best: return this many hypotheses, <= beam (currently only 1)
        '''
        def __init__(self, config_path: str, n_best: int = 1):
            seed = 42
            torch.manual_seed(seed)
            cfg = load_config(config_path)
            args = parse_global_args(cfg, rank=0, mode="translate")
            self.args = args._replace(test=args.test._replace(n_best=n_best))
            # build model
            self.model, _, _, self.test_data = prepare(self.args, rank=0, mode="translate")

        def _translate_data(self):
            _, _, hypotheses, trg_tokens, trg_scores, _ = predict(
                model=self.model,
                data=self.test_data,
                compute_loss=False,
                device=self.args.device,
                rank=0,
                n_gpu=self.args.n_gpu,
                normalization="none",
                num_workers=self.args.num_workers,
                args=self.args.test,
                autocast=self.args.autocast,
            )
            return hypotheses, trg_tokens, trg_scores

        def translate(self, sentence) -> list:
            '''
            Translate the given sentence.

            :param sentence: Sentence to be translated
            :return:
            - translations: (list of str) possible translations of the sentence.
            '''
            self.test_data.set_item(sentence.strip())
            translations, _, _ = self._translate_data()
            assert len(translations) == len(self.test_data) * self.args.test.n_best
            self.test_data.reset_cache()
            return translations

    # Load model
    config_path = "/path/to/lean_model/config_local.yaml" # Change this to actual path
    model = JoeyNMTModel(config_path)

    # Translate
    sentence = "This is an example sentence."
    translations = model.translate(sentence)
    print(translations)
    """
    with open(Path(lean_model_dir) / "README.md", 'w') as f:
        f.write(model_card)
def push_to_hub(files, HF_REPO_NAME):
    api = HfApi()
    for file_path in files:
        print(file_path.name)
        print(str(file_path))
        api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.name,
        repo_id=HF_REPO_NAME,
        )

def main():
    # Directories
    data_dir = "/experiment/data"
    model_dir = "/experiment/model"
    HF_REPO_NAME = "Koleshjr/trial"
    lean_model_dir = "/experiment/lean-model"
    # Clean and prepare data
    chars_to_remove_regex = r'[^\w\s]'
    src_lang = "dyu"
    trg_lang = "fr"

    dataset = load_dataset("dyu_fr_dataset")
    dataset = dataset.map(lambda batch: clean_text(batch, src_lang, trg_lang, chars_to_remove_regex))
    dataset.save_to_disk(data_dir)

    # Create and save configuration
    config = create_config(data_dir, model_dir)
    config = append_training_testing_config(config, data_dir, model_dir)
    save_and_train(config, data_dir)

    # Update configuration with model info
    update_config_with_model_info(model_dir, data_dir)

    # Copy necessary files
    src_files = [
        (Path(model_dir) / "sp.model", Path(lean_model_dir) / "sp.model"),
        (Path(data_dir) / "vocab.txt", Path(lean_model_dir) / "vocab.txt"),
        (Path(model_dir) / "config.yaml", Path(lean_model_dir) / "config.yaml"),
        (Path(model_dir) / "best.ckpt", Path(lean_model_dir) / "best.ckpt"),
        # Add other files as necessary
    ]
    copy_files(src_files, model_dir)

    # Create model card
    create_model_card(HF_REPO_NAME, lean_model_dir)

    # Push files to Hugging Face Hub
    files_to_push = [
        Path(lean_model_dir) / "sp.model",
        Path(lean_model_dir) / "vocab.txt",
        Path(lean_model_dir) / "README.md",
        Path(lean_model_dir) / "config.yaml",
        Path(lean_model_dir) / "best.ckpt",
        # Add other files as necessary
    ]
    push_to_hub(files_to_push, HF_REPO_NAME)
