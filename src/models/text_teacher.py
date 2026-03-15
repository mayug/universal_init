"""Text teacher models for cross-modal distillation.

Provides frozen text encoders whose embeddings serve as distillation targets
for audio student models. The audio student processes spectrograms while
these teachers process the corresponding text captions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CLIPTextTeacher(nn.Module):
    """Frozen CLIP text encoder.

    Uses the text tower of CLIP ViT-L/14 to produce L2-normalized text
    embeddings from captions. The model has never seen or heard audio —
    testing whether its cross-modal geometry transfers to audio is the
    core experiment.

    Args:
        model_name: HuggingFace model identifier for CLIP.
        device: Device to place the model on.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device

        from transformers import CLIPModel, CLIPTokenizerFast

        print(f"Loading CLIP text encoder: {model_name}")
        clip_model = CLIPModel.from_pretrained(model_name)
        self.text_model = clip_model.text_model.to(device)
        self.text_projection = clip_model.text_projection.to(device)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)

        # Detect embedding dimension
        self.embed_dim = clip_model.text_projection.out_features
        print(f"  Embed dim: {self.embed_dim}")

        # Freeze all parameters
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.text_projection.eval()
        for param in self.text_projection.parameters():
            param.requires_grad = False

        # Clean up the full CLIP model (we only kept text parts)
        del clip_model

    def tokenize(self, captions: List[str]) -> dict:
        """Tokenize a list of text captions.

        Args:
            captions: List of caption strings.

        Returns:
            Tokenizer output dict with input_ids and attention_mask on device.
        """
        tokens = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    @torch.no_grad()
    def forward(self, text_tokens: dict) -> torch.Tensor:
        """Encode tokenized text to L2-normalized embeddings.

        Args:
            text_tokens: Dict with 'input_ids' and 'attention_mask' tensors.

        Returns:
            embeddings: L2-normalized text embeddings [B, embed_dim].
        """
        outputs = self.text_model(
            input_ids=text_tokens["input_ids"],
            attention_mask=text_tokens["attention_mask"],
        )
        # CLIP uses the [EOS] token embedding, then projects
        # pooler_output is the [EOS] token representation
        pooled = outputs.pooler_output
        projected = self.text_projection(pooled)
        return F.normalize(projected, p=2, dim=-1)

    @torch.no_grad()
    def encode(self, captions: List[str]) -> torch.Tensor:
        """Convenience method: tokenize and encode in one call.

        Args:
            captions: List of caption strings.

        Returns:
            embeddings: L2-normalized text embeddings [B, embed_dim].
        """
        tokens = self.tokenize(captions)
        return self.forward(tokens)


class SentenceBERTTeacher(nn.Module):
    """Frozen Sentence-BERT text encoder.

    Uses all-mpnet-base-v2 to produce L2-normalized text embeddings.
    This is a unimodal text encoder — comparing with CLIP tests whether
    cross-modal pretraining geometry (CLIP) is better than unimodal (SBERT).

    Args:
        model_name: sentence-transformers model identifier.
        device: Device to place the model on.
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: str = "cuda",
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device

        from sentence_transformers import SentenceTransformer

        print(f"Loading Sentence-BERT: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)

        # Detect embedding dimension
        self.embed_dim = self.model.get_sentence_embedding_dimension()
        print(f"  Embed dim: {self.embed_dim}")

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, captions: List[str]) -> torch.Tensor:
        """Encode captions to L2-normalized embeddings.

        Args:
            captions: List of caption strings.

        Returns:
            embeddings: L2-normalized text embeddings [B, embed_dim].
        """
        # sentence-transformers returns numpy by default
        embeddings = self.model.encode(
            captions,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.to(self.device)
