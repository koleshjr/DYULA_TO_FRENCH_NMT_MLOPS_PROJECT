import argparse
import re
from kserve import (
    InferOutput,
    InferRequest,
    InferResponse,
    Model,
    ModelServer,
    model_server
)

from kserve.utils.utils import generate_uuid
from transformers import pipeline,  T5Tokenizer
MODEL_DIR = "/app/saved_model"
CHARS_TO_REMOVE_REGEX = '[!"&\(\),-./:;=?+.\n\[\]]'
PREFIX = "Translate the following sentence from Dyula to French: "  # Model's inference command

def clean_translation(translation):
    CHARS_TO_REMOVE_REGEX = '[!"&\(\),-./:;=?+.\n\[\]]'
    return re.sub(CHARS_TO_REMOVE_REGEX, " ", translation.lower()).strip()

class MyModel(Model):
    """Kserve Inference Implementation of Model"""
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.tokenizer = None
        self.pipe_ft = None
        self.ready = False 
        self.load()
    
    def load(self):
        """Reconstitue Model From Disk"""
        repository_id = f"{MODEL_DIR}/Koleshjrflan-t5-base-finetuned-translation-v5"
        self.tokenizer = T5Tokenizer.from_pretrained(repository_id)
        self.pipe_ft = pipeline("translation", model = repository_id, max_length=self.tokenizer.model_max_length, device_map="auto")
        self.ready = True   

    def preprocess(self, payload: InferRequest, *args, **kwargs) -> str:
        """Preprocess inference request."""
        # Clean input sentence and add prefix
        raw_data = payload.inputs[0].data[0]
        prepared_data = f"{PREFIX}{clean_translation(raw_data)}"
        return prepared_data

    def predict(self, data: str, *args, **kwargs) -> InferResponse:
        out = self.pipe_ft(data)
        translation = out[0]['translation_text']
        response_id = generate_uuid()
        infer_output = InferOutput(
            name="output-0", shape=[1], datatype="STR", data=[translation]
        )
        infer_response = InferResponse(
            model_name=self.name, infer_outputs=[infer_output], response_id=response_id
        )
        return infer_response
    

parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--model_name", default="model", help="The name that the model is served under."
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MyModel(args.model_name)
    ModelServer().start([model])      



