# load the model
import torch
from joeynmt.config import load_config, parse_global_args
from joeynmt.prediction import predict, prepare

class JoeyNMTModel:
    """
    JoeyNMTModel which load JoeyNMT model for inference.

    :param config_path: Path to YAML config file
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    """
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
        """
        Translate the given sentence.

        :param sentence: Sentence to be translated
        :return:
        - translations: (list of str) possible translations of the sentence.
        """
        self.test_data.set_item(sentence.strip())
        translations, _, _ = self._translate_data()
        assert len(translations) == len(self.test_data) * self.args.test.n_best
        self.test_data.reset_cache()
        return translations
config_path = "saved_model/config.yaml" # Change this to the path to your model congig file
model = JoeyNMTModel(config_path=config_path, n_best=1)

if __name__ == "__main__":
    for example in [{'dyu': 'i tɔgɔ bi cogodɔ', 'fr': 'tu portes un nom de fantaisie'},
    {'dyu': 'puɛn saba fɔlɔ', 'fr': 'trois points d’avance'},
    {'dyu': 'tile bena', 'fr': 'le soleil s’est couché'}]:
        print(model.translate(sentence=example['dyu']))
        print(example['fr'])
        print("-"* 100)