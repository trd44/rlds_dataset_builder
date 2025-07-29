from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class HanoiDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

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
                            doc='Robot EEF state (6D pose, 2D gripper).',
                        )
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
            'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            raw = np.load(episode_path, allow_pickle=True)
            print("DEBUG: type(raw)=", type(raw))
            # Unwrap the dict if needed
            if isinstance(raw, dict):
                episode_dict = raw
            elif isinstance(raw, np.ndarray):
                # np.save(dict) -> array([dict], dtype=object) or array(dict, dtype=object)
                try:
                    episode_dict = raw.item()
                except Exception:
                    episode_dict = raw[0]
            else:
                raise ValueError(f"Can't parse type={type(raw)} from {episode_path}")

            # --- Added inspection and summary code ---
            print(f"Inspecting episode: {episode_path}")
            # Check if top-level has 'steps' key or is just a list/array
            if isinstance(episode_dict, dict) and 'steps' in episode_dict:
                steps = episode_dict['steps']
                print(f"'steps' key found at top level with length: {len(steps)}")
            elif isinstance(episode_dict, (list, np.ndarray)):
                steps = episode_dict
                print(f"Top-level is a list/array with length: {len(steps)}")
            else:
                steps = None
                print("Warning: No 'steps' key and top-level is not a list/array.")

            expected_keys = {'observation', 'action', 'discount', 'reward', 'is_first', 'is_last', 'is_terminal', 'language_instruction'}
            all_keys_found = set()
            missing_keys_any_step = False

            if steps is not None:
                for i, step in enumerate(steps):
                    if i < 5:
                        print(f"Step {i}:")
                        if isinstance(step, dict):
                            step_keys = set(step.keys())
                            all_keys_found.update(step_keys)
                            print(f"  Keys: {list(step_keys)}")
                            # Observation keys and their sub-keys
                            if 'observation' in step and isinstance(step['observation'], dict):
                                obs = step['observation']
                                print(f"  observation sub-keys:")
                                for k, v in obs.items():
                                    if isinstance(v, np.ndarray):
                                        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                                    else:
                                        print(f"    {k}: type={type(v)}")
                            # For other keys, print type/value for first 2 steps
                            for k in step_keys:
                                if k != 'observation' and i < 2:
                                    val = step[k]
                                    if isinstance(val, np.ndarray):
                                        print(f"  {k}: shape={val.shape}, dtype={val.dtype}")
                                    else:
                                        print(f"  {k}: type={type(val)}, value={val}")
                            # Check for missing keys in this step
                            missing_keys = expected_keys - step_keys
                            if missing_keys:
                                print(f"  Warning: Missing keys in step {i}: {missing_keys}")
                                missing_keys_any_step = True
                        else:
                            print(f"  Warning: Step {i} is not a dict, but {type(step)}")
                    else:
                        # For steps beyond first 5, just accumulate keys
                        if isinstance(step, dict):
                            all_keys_found.update(step.keys())
                        else:
                            print(f"  Warning: Step {i} is not a dict, but {type(step)}")
            else:
                print("No steps to inspect.")

            if missing_keys_any_step:
                print("Warning: Some steps are missing expected keys!")

            print("Summary of all unique keys found across steps:")
            for key in sorted(all_keys_found):
                print(f"  {key}")

            # --- End of added inspection and summary code ---

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
