import os
import json
import logging
from pathlib import Path

import torch

from huggingface_hub import hf_hub_download

from tokan.bigvgan import bigvgan
from tokan.bigvgan.env import AttrDict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelManager:
    """Handles model loading, downloading and caching."""

    BASE_DIR = "pretrained_models"

    MODELS_CONFIG = {
        # Direct download from URLs
        "hubert": {
            "url": "https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt",
            "local_path": os.path.join(BASE_DIR, "hubert", "hubert_large_ll60k.pt"),
        },
        # Download from HuggingFace
        "hubert_km": {
            "repo_id": "Piping/TokAN-Legacy",
            "revision": "main",
            "file_path": "hubert_km/hubert_km_libritts_l17_1000.pt",
            "local_dir": BASE_DIR,
        },
        "token_to_token": {
            "repo_id": "Piping/TokAN-Legacy",
            "revision": "main",
            "file_path": "token_to_token/tokan-t2t-base-paper/model.pt",
            "local_dir": BASE_DIR,
            "auxiliary_files": ["dict.src.txt", "dict.tgt.txt", "dict.aux.txt"],
        },
        "token_to_mel_v1": {  # Regression-based duration predictor
            "repo_id": "Piping/TokAN-Legacy",
            "revision": "main",
            "file_path": "token_to_mel/tokan-t2m-v1-paper/model.ckpt",
            "local_dir": BASE_DIR,
        },
        "token_to_mel_v2": {  # Flow-matching based duration predictor
            "repo_id": "Piping/TokAN-Legacy",
            "revision": "main",
            "file_path": "token_to_mel/tokan-t2m-v2-paper/model.ckpt",
            "local_dir": BASE_DIR,
        },
    }

    @classmethod
    def ensure_model_available(cls, model_key: str) -> str:
        """Ensure model and its auxiliary files are available, downloading if necessary."""
        if model_key not in cls.MODELS_CONFIG:
            raise ValueError(f"Unknown model key: {model_key}")

        config = cls.MODELS_CONFIG[model_key]
        if "local_path" in config:
            local_dir, file_path = "", ""
            local_path = config["local_path"]
        else:
            local_dir = config.get("local_dir", cls.BASE_DIR)
            file_path = config["file_path"]
            local_path = os.path.join(local_dir, file_path)

        # Create directory structure
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists
        if Path(local_path).exists():
            logger.info(f"Model already exists at {local_path}")
            return local_path

        # Handle direct URL downloads
        if "url" in config:
            logger.info(f"Downloading from direct URL: {config['url']}")
            try:
                cls.download_file_from_url(config["url"], local_path)
            except Exception as e:
                logger.error(f"Error downloading file from URL: {e}")
                raise

        # Handle HuggingFace Hub downloads
        elif "repo_id" in config and "file_path" in config:
            repo_id = config["repo_id"]
            revision = config.get("revision", "main")

            logger.info(f"Downloading {file_path} from {repo_id}")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    revision=revision,
                    local_dir=local_dir,  # Use the parent directory of the target file
                    local_dir_use_symlinks=False,
                )
                logger.info(f"Downloaded to {downloaded_path}")
            except Exception as e:
                logger.error(f"Error downloading file from Hugging Face: {e}")
                raise

            # Download auxiliary files if any
            if "auxiliary_files" in config:
                for aux_file in config["auxiliary_files"]:
                    aux_local_path = Path(local_path).parent / aux_file
                    aux_file_path = str(Path(file_path).parent / aux_file)
                    if not aux_local_path.exists():
                        logger.info(f"Downloading auxiliary file {aux_file_path}")
                        try:
                            hf_hub_download(
                                repo_id=repo_id,
                                filename=aux_file_path,
                                revision=revision,
                                local_dir=local_dir,
                                local_dir_use_symlinks=False,
                            )
                        except Exception as e:
                            logger.warning(f"Could not download auxiliary file {aux_file}: {e}")
        else:
            raise ValueError(f"Config for {model_key} must contain either 'url' or both 'repo_id' and 'file_path'")

        return local_path

    @staticmethod
    def download_file_from_url(url: str, destination: str) -> None:
        """Download a file from URL to destination."""
        import requests
        from tqdm.auto import tqdm

        logger.info(f"Downloading {url} to {destination}")

        # Make sure the destination directory exists
        Path(destination).parent.mkdir(parents=True, exist_ok=True)

        # Stream the download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        total_size = int(response.headers.get("content-length", 0))

        with open(destination, "wb") as file:
            with tqdm(
                desc=Path(destination).name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)


def load_bigvgan(tag_or_ckpt: str, device: str = "cuda"):
    """
    Load BigVGAN model from a tag or checkpoint.

    Args:
        tag_or_ckpt: Either a Hugging Face model tag (e.g., 'nvidia/bigvgan_22khz_80band')
                    or a path to a local checkpoint file
        device: Device to load the model onto

    Returns:
        Loaded BigVGAN model
    """
    logger.info(f"Loading BigVGAN model: {tag_or_ckpt}")

    # Check if it's a Hugging Face model tag or local path
    is_hf_tag = tag_or_ckpt.startswith("nvidia/") and not os.path.exists(tag_or_ckpt)

    if is_hf_tag:
        logger.info(f"Detected Hugging Face model tag, loading from hub: {tag_or_ckpt}")
        try:
            vocoder = bigvgan.BigVGAN.from_pretrained(tag_or_ckpt)
        except Exception as e:
            logger.error(f"Failed to load from Hugging Face: {e}")
            raise RuntimeError(f"Could not load BigVGAN model from Hugging Face tag '{tag_or_ckpt}': {e}")
    else:
        # Assume it's a local checkpoint
        logger.info(f"Loading BigVGAN from local checkpoint: {tag_or_ckpt}")

        if not os.path.exists(tag_or_ckpt):
            raise FileNotFoundError(f"Local checkpoint file not found: {tag_or_ckpt}")

        exp_dir = os.path.dirname(tag_or_ckpt)
        config_path = os.path.join(exp_dir, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path) as f:
                data = f.read()
            json_config = json.loads(data)
            h = AttrDict(json_config)

            vocoder = bigvgan.BigVGAN(h)

            state_dict = torch.load(tag_or_ckpt, map_location="cpu")
            vocoder.load_state_dict(state_dict["generator"])

        except Exception as e:
            logger.error(f"Failed to load local checkpoint: {e}")
            raise RuntimeError(f"Could not load BigVGAN model from local checkpoint '{tag_or_ckpt}': {e}")

    vocoder = vocoder.eval().to(device)
    vocoder.remove_weight_norm()

    for param in vocoder.parameters():
        param.requires_grad = False

    return vocoder


if __name__ == "__main__":
    # Download the pretrained models when running this script

    for model_key in ModelManager.MODELS_CONFIG.keys():
        try:
            model_path = ModelManager.ensure_model_available(model_key)
            print(f"Model and auxiliary files for '{model_key}' are available at: {model_path}")
        except Exception as e:
            print(f"Error ensuring model availability for '{model_key}': {e}")

    _ = load_bigvgan("nvidia/bigvgan_22khz_80band", "cpu")
