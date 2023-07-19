"""satellite dataset."""

import tensorflow_datasets.public_api as tfds
from . import satellite_dataset_builder


class SatelliteTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for satellite dataset."""
  # TODO(satellite):
  DATASET_CLASS = satellite_dataset_builder.My_Satellite_train
  SPLITS = {
      'train': 7140,  # Number of fake train example
      'test': 0,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
