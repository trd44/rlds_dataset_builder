from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class Hanoi50(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Assembly Line Sorting dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot Proprio state (7D Joint angles, 1D gripper).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot EEF action.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    # 'language_embedding': tfds.features.Tensor(
                    #     shape=(512,),
                    #     dtype=np.float32,
                    #     doc='Kona language embedding. '
                    #         'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    # ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/episode_*.npy'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
            # 'test': self._generate_examples(path='data/test/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            raw = np.load(episode_path, allow_pickle=True)

            seq = None
            if isinstance(raw, dict):
                episode_dict = raw
            elif isinstance(raw, np.ndarray):
                try:
                    episode_dict = raw.item()
                except Exception:
                    episode_dict = raw[0]
            else:
                raise ValueError(f"Can't parse type={type(raw)} from {episode_path}")

            seq = episode_dict.get('steps')
            if seq is None:
                raise ValueError(f"Missing 'steps' in {episode_path}")

            if not isinstance(seq, (list, np.ndarray)):
                raise TypeError(f"After loading, expected sequence, got {type(seq)}")

            episode = []
            for i, step in enumerate(seq):
                if not isinstance(step, dict):
                    raise TypeError(f"Expected dict step, got {type(step)} in {episode_path}")
                # compute Kona language embedding
                # language_embedding = self._embed([step['language_instruction']])[0].numpy()

                episode.append({
                    'observation': {
                        'image': step['observation']['image'],
                        'wrist_image': step['observation']['wrist_image'],
                        'state': step['observation']['state'],
                    },
                    'action': step['action'],
                    'discount': 1.0,
                    'reward': float(i == (len(seq) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(seq) - 1),
                    'is_terminal': i == (len(seq) - 1),
                    'language_instruction': step['language_instruction'],
                    # 'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
