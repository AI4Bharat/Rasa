import json
import os
from typing import Any, Dict, List, Union

import fsspec
import numpy as np
import torch
from coqpit import Coqpit

from TTS.config import get_from_config_or_model_args_with_default
from TTS.tts.utils.managers import EmbeddingManager


class CommonManager(EmbeddingManager):
    """Manage the labels for multi-label 🐸TTS models. Load a datafile and parse the information
    in a way that can be queried by label or clip.

    There are 3 different scenarios considered:

    1. Models using label embedding layers. The datafile only maps label names to ids used by the embedding layer.
    2. Models using d-vectors. The datafile includes a dictionary in the following format.

    ::

        {
            'clip_name.wav':{
                'name': 'labelA',
                'embedding'[<d_vector_values>]
            },
            ...
        }

    Examples:
        >>> # load audio processor and label encoder
        >>> ap = AudioProcessor(**config.audio)
        >>> manager = CommonManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)
        >>> # load a sample audio and compute embedding
        >>> waveform = ap.load_wav(sample_wav_path)
        >>> mel = ap.melspectrogram(waveform)
        >>> d_vector = manager.compute_embeddings(mel.T)
    """

    def __init__(
        self,
        data_items: List[List[Any]] = None,
        d_vectors_file_path: str = "",
        label_id_file_path: str = "",
        encoder_model_path: str = "",
        encoder_config_path: str = "",
        use_cuda: bool = False,
    ):
        super().__init__(
            embedding_file_path=d_vectors_file_path,
            id_file_path=label_id_file_path,
            encoder_model_path=encoder_model_path,
            encoder_config_path=encoder_config_path,
            use_cuda=use_cuda,
        )

        if data_items:
            self.set_ids_from_data(data_items, parse_key="label_name")

    @property
    def num_labels(self):
        return len(self.ids)

    @property
    def label_names(self):
        return list(self.ids.keys())

    def get_labels(self) -> List:
        return self.ids

    @staticmethod
    def init_from_config(config: "Coqpit", samples: Union[List[List], List[Dict]] = None) -> "CommonManager":
        """Initialize a label manager from config

        Args:
            config (Coqpit): Config object.
            samples (Union[List[List], List[Dict]], optional): List of data samples to parse out the label names.
                Defaults to None.

        Returns:
            labelEncoder: label encoder object.
        """
        label_manager = None
        if get_from_config_or_model_args_with_default(config, "use_label_embedding", False):
            if samples:
                label_manager = CommonManager(data_items=samples)
            if get_from_config_or_model_args_with_default(config, "label_file", None):
                label_manager = CommonManager(
                    label_id_file_path=get_from_config_or_model_args_with_default(config, "label_file", None)
                )
            if get_from_config_or_model_args_with_default(config, "labels_file", None):
                label_manager = CommonManager(
                    label_id_file_path=get_from_config_or_model_args_with_default(config, "labels_file", None)
                )

        if get_from_config_or_model_args_with_default(config, "use_d_vector_file", False):
            label_manager = CommonManager()
            if get_from_config_or_model_args_with_default(config, "labels_file", None):
                label_manager = CommonManager(
                    d_vectors_file_path=get_from_config_or_model_args_with_default(config, "label_file", None)
                )
            if get_from_config_or_model_args_with_default(config, "d_vector_file", None):
                label_manager = CommonManager(
                    d_vectors_file_path=get_from_config_or_model_args_with_default(config, "d_vector_file", None)
                )
        return label_manager
    
    @staticmethod
    def init_from_config_with_label_key(config: "Coqpit", samples: Union[List[List], List[Dict]] = None, label_key=None) -> "CommonManager":
        """Initialize a label manager from config

        Args:
            config (Coqpit): Config object.
            samples (Union[List[List], List[Dict]], optional): List of data samples to parse out the label names.
                Defaults to None.

        Returns:
            labelEncoder: label encoder object.
        """
        if label_key is None:
            raise ValueError("Must provide label key to initialize CommonManager. For example, 'language' or 'emotion'.")
        label_manager = None
        if get_from_config_or_model_args_with_default(config, f"use_{label_key}_embedding", False):
            if samples:
                label_manager = CommonManager(data_items=samples)
            if get_from_config_or_model_args_with_default(config, f"{label_key}_file", None):
                label_manager = CommonManager(
                    label_id_file_path=get_from_config_or_model_args_with_default(config, f"{label_key}_file", None)
                )
            if get_from_config_or_model_args_with_default(config, f"{label_key}s_file", None):
                label_manager = CommonManager(
                    label_id_file_path=get_from_config_or_model_args_with_default(config, f"{label_key}s_file", None)
                )

        if get_from_config_or_model_args_with_default(config, "use_d_vector_file", False):
            label_manager = CommonManager()
            if get_from_config_or_model_args_with_default(config, f"{label_key}s_file", None):
                label_manager = CommonManager(
                    d_vectors_file_path=get_from_config_or_model_args_with_default(config, f"{label_key}_file", None)
                )
            if get_from_config_or_model_args_with_default(config, "d_vector_file", None):
                label_manager = CommonManager(
                    d_vectors_file_path=get_from_config_or_model_args_with_default(config, "d_vector_file", None)
                )
        return label_manager


def _set_file_path(path):
    """Find the labels.json under the given path or the above it.
    Intended to band aid the different paths returned in restored and continued training."""
    path_restore = os.path.join(os.path.dirname(path), "labels.json")
    path_continue = os.path.join(path, "labels.json")
    fs = fsspec.get_mapper(path).fs
    if fs.exists(path_restore):
        return path_restore
    if fs.exists(path_continue):
        return path_continue
    raise FileNotFoundError(f" [!] `labels.json` not found in {path}")


def load_label_mapping(out_path):
    """Loads label mapping if already present."""
    if os.path.splitext(out_path)[1] == ".json":
        json_file = out_path
    else:
        json_file = _set_file_path(out_path)
    with fsspec.open(json_file, "r") as f:
        return json.load(f)


def save_label_mapping(out_path, label_mapping):
    """Saves label mapping if not yet present."""
    if out_path is not None:
        labels_json_path = _set_file_path(out_path)
        with fsspec.open(labels_json_path, "w") as f:
            json.dump(label_mapping, f, indent=4)


def get_label_manager(c: Coqpit, data: List = None, restore_path: str = None, out_path: str = None) -> CommonManager:
    """Initiate a `CommonManager` instance by the provided config.

    Args:
        c (Coqpit): Model configuration.
        restore_path (str): Path to a previous training folder.
        data (List): Data samples used in training to infer labels from. It must be provided if label embedding
            layers is used. Defaults to None.
        out_path (str, optional): Save the generated label IDs to a output path. Defaults to None.

    Returns:
        CommonManager: initialized and ready to use instance.
    """
    label_manager = CommonManager()
    if c.use_label_embedding:
        if data is not None:
            label_manager.set_ids_from_data(data, parse_key="label_name")
        if restore_path:
            labels_file = _set_file_path(restore_path)
            # restoring label manager from a previous run.
            if c.use_d_vector_file:
                # restore label manager with the embedding file
                if not os.path.exists(labels_file):
                    print("WARNING: labels.json was not found in restore_path, trying to use CONFIG.d_vector_file")
                    if not os.path.exists(c.d_vector_file):
                        raise RuntimeError(
                            "You must copy the file labels.json to restore_path, or set a valid file in CONFIG.d_vector_file"
                        )
                    label_manager.load_embeddings_from_file(c.d_vector_file)
                label_manager.load_embeddings_from_file(labels_file)
            elif not c.use_d_vector_file:  # restor label manager with label ID file.
                label_ids_from_data = label_manager.ids
                label_manager.load_ids_from_file(labels_file)
                assert all(
                    label in label_manager.ids for label in label_ids_from_data
                ), " [!] You cannot introduce new labels to a pre-trained model."
        elif c.use_d_vector_file and c.d_vector_file:
            # new label manager with external label embeddings.
            label_manager.load_embeddings_from_file(c.d_vector_file)
        elif c.use_d_vector_file and not c.d_vector_file:
            raise "use_d_vector_file is True, so you need pass a external label embedding file."
        elif c.use_label_embedding and "labels_file" in c and c.labels_file:
            # new label manager with label IDs file.
            label_manager.load_ids_from_file(c.labels_file)

        if label_manager.num_labels > 0:
            print(
                " > label manager is loaded with {} labels: {}".format(
                    label_manager.num_labels, ", ".join(label_manager.ids)
                )
            )

        # save file if path is defined
        if out_path:
            out_file_path = os.path.join(out_path, "labels.json")
            print(f" > Saving `labels.json` to {out_file_path}.")
            if c.use_d_vector_file and c.d_vector_file:
                label_manager.save_embeddings_to_file(out_file_path)
            else:
                label_manager.save_ids_to_file(out_file_path)
    return label_manager


def get_label_balancer_weights(items: list):
    label_names = np.array([item["label_name"] for item in items])
    unique_label_names = np.unique(label_names).tolist()
    label_ids = [unique_label_names.index(l) for l in label_names]
    label_count = np.array([len(np.where(label_names == l)[0]) for l in unique_label_names])
    weight_label = 1.0 / label_count
    dataset_samples_weight = np.array([weight_label[l] for l in label_ids])
    # normalize
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    return torch.from_numpy(dataset_samples_weight).float()
