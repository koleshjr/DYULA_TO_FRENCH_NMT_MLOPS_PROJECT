---
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
    repo_id="Koleshjr/dyu-fr-joeynmt-316-epochs_2_layers_5heads_128_300_plateau_2000_7_24_21_44_relu",
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
config_path = "/path/to/lean_model/config_local.yaml" # Change this to the path to your model congig file
model = JoeyNMTModel(config_path=config_path, n_best=1)

# Translate
model.translate(sentence="i tɔgɔ bi cogodɔ")
```

## Training procedure

### Training hyperparameters

More information needed

### Training results

More information needed

### Framework versions

- JoeyNMT 2.3.0
- Torch 2.4.0+cu121

