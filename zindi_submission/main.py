"""
KServe endpoint for JoeyNMT translation model.

This module provides a KServe-compatible endpoint for a JoeyNMT translation model,
allowing for easy deployment and inference of the model.
"""

import re
import argparse
from typing import List, Any
import torch
from joeynmt.prediction import predict, prepare
from joeynmt.config import load_config, parse_global_args
from kserve.utils.utils import generate_uuid
from kserve import Model, ModelServer, model_server, InferRequest, InferOutput, InferResponse

MODEL_CONFIG_PATH = "/app/saved_model/config.yaml"
CHARS_TO_REMOVE_COMPILED = re.compile(r'[!"&(),\-./:;=?+.\n\[\]]')

def clean_text(text: str) -> str:
    """
    Clean input text by removing special characters and converting to lower case.
    """
    return CHARS_TO_REMOVE_COMPILED.sub(" ", text.lower()).strip()

class JoeyNMTModelDyuFr:
    """JoeyNMTModelDyuFr which loads JoeyNMT model for inference."""

    def __init__(self, config_path: str, n_best: int = 1) -> None:
        """
        Initialize a JoeyNMTModelDyuFr instance with a loaded model for inference.

        Parameters:
        - config_path (str): Path to the YAML configuration file for the model.
        - n_best (int, optional): Number of hypotheses to return. Default is 1.
        """
        seed = 42
        torch.manual_seed(seed)
        cfg = load_config(config_path)
        global_args = parse_global_args(cfg, rank=0, mode="translate")
        self.args = global_args._replace(test=global_args.test._replace(n_best=n_best))
        # build model
        self.model, _, _, self.test_data = prepare(self.args, rank=0, mode="translate")

    def translate_data(self) -> List[str]:
        """
        Translate the data using the loaded model and return the hypotheses.

        Returns:
        - hypotheses: A list of strings representing the translated hypotheses.
        """
        _, _, hypotheses, _, _, _ = predict(
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
        return hypotheses

    def translate(self, sentence: str) -> List[str]:
        """
        Translate the given sentence.

        :param sentence: Sentence to be translated
        :return: A list of possible translations of the sentence.
        """
        self.test_data.set_item(sentence.strip())
        translations = self.translate_data()
        self.test_data.reset_cache()
        return translations

class MyModel(Model):
    """Model class for KServe endpoint."""

    def __init__(self, name: str):
        """
        Initialize a MyModel instance with the provided name.

        Parameters:
        - name (str): The name that the model will be served under.
        """
        super().__init__(name)
        self.name = name
        self.model = None
        self.ready = False
        self.load()

    def load(self):
        """Load the JoeyNMT model for translation."""
        if not self.model:
            self.model = JoeyNMTModelDyuFr(config_path=MODEL_CONFIG_PATH, n_best=1)
            self.ready = True

    async def preprocess(self, payload: InferRequest, headers: Any = None) -> List[str]:
        """
        Preprocess the input data for translation.

        Parameters:
        - payload (InferRequest): The input payload containing the data to be preprocessed.
        - headers (Any, optional): Request headers. Defaults to None.

        Returns:
        - cleaned_texts (List[str]): A list of cleaned text strings.
        """
        infer_inputs: List[str] = payload.inputs[0].data
        print(f"** infer_input ({type(infer_inputs)}): {infer_inputs}")

        cleaned_texts: List[str] = [clean_text(i) for i in infer_inputs]
        print(f"** cleaned_text ({type(cleaned_texts)}): {cleaned_texts}")
        return cleaned_texts

    async def predict(self, payload: List[str], headers: Any = None) -> InferResponse:
        """
        Perform translation on the input data using the loaded model.

        Parameters:
        - payload (List[str]): A list of strings representing the input sentences to be translated.
        - headers (Any, optional): Request headers. Defaults to None.

        Returns:
        - infer_response (InferResponse): An InferResponse object containing the translated results.
        """
        response_id = generate_uuid()
        results: List[str] = [self.model.translate(sentence=s)[0] for s in payload]
        print(f"** result ({type(results)}): {results}")

        infer_output = InferOutput(name="output-0",
                                shape=[len(results)],
                                datatype="STR",
                                data=results)
        return InferResponse(model_name=self.name,
                            infer_outputs=[infer_output],
                            response_id=response_id)

parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--custom_model_name",
    default="model",
    help="The name that the model is served under."
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MyModel(args.model_name)
    ModelServer().start([model])
