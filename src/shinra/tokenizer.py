"""Local-only SHINRA tokenizer contract. Never fetches Hub artifacts."""

from pathlib import Path
import hashlib
import json


class ShinraTokenizer:
    def __init__(self, model_path, contract_path, config):
        import sentencepiece as spm
        from sentencepiece import sentencepiece_model_pb2

        data = Path(model_path).read_bytes()
        contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
        if hashlib.sha256(data).hexdigest() != contract["sha256"]:
            raise ValueError("Tokenizer hash mismatch")
        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString(data)
        if proto.trainer_spec.model_type != 1 or not proto.trainer_spec.byte_fallback:
            raise ValueError("SHINRA requires Unigram with byte fallback")
        self.processor = spm.SentencePieceProcessor(model_proto=data)
        if self.processor.vocab_size() != config.vocab_size:
            raise ValueError("Tokenizer vocabulary differs from model contract")
        self.controls = contract["controls"]
        if len(set(self.controls.values())) != len(self.controls):
            raise ValueError("Control IDs must be unique")
        if any(not 0 <= value < config.vocab_size for value in self.controls.values()):
            raise ValueError("Control ID out of range")

    def encode(self, text):
        return self.processor.encode(text, out_type=int)

    def decode(self, ids):
        return self.processor.decode(ids)

    def fertility(self, texts):
        tokens = sum(len(self.encode(text)) for text in texts)
        byte_count = sum(len(text.encode("utf-8")) for text in texts)
        characters = sum(len(text) for text in texts)
        return {
            "tokens": tokens,
            "tokens_per_byte": tokens / max(1, byte_count),
            "tokens_per_character": tokens / max(1, characters),
        }
