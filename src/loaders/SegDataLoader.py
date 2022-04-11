import os
from abc import ABC, abstractmethod
from glob import glob
from typing import Any, Dict

import numpy as np
import numpy.random as random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

val = [
    "ec09336a6f_06BA0AF311OPENPIPELINE",
    "679850f980_27920CBE78OPENPIPELINE",
    "c8a7031e5f_32156F5DC2INSPIRE",
    "6b82bcd67b_2EBB40A325OPENPIPELINE",
    "cc4b443c7d_A9CBEF2C97INSPIRE",
    "12c3372a95_7EF127EDCFINSPIRE",
    "941cb687d3_48FE7F729BINSPIRE",
    "42ab9f9e27_3CB2E8FC73INSPIRE",
    "264c36d368_C988C95F03INSPIRE",
    "954a8c814c_267994885AINSPIRE",
    "ea607f191d_582C2A2F47OPENPIPELINE",
    "600023a2df_F4A3C2E777INSPIRE",
    "57426ebe1e_84B52814D2OPENPIPELINE",
    "cd5a0d3ce4_2F98B8FC82INSPIRE",
    "3731e901b0_9464BAFE8AOPENPIPELINE",
    "f0c32df5a8_0406E6C238OPENPIPELINE",
    "1476907971_CHADGRISMOPENPIPELINE",
    "97c4dd388d_4C51642B86OPENPIPELINE",
    "f78c4e5748_3572E1D9BBOPENPIPELINE",
    "a11d963a7d_EF73EE9CCDOPENPIPELINE",
    "aef48b9aca_0226FDD487OPENPIPELINE",
    "9170479165_625EDFBAB6OPENPIPELINE",
    "3bb457cde8_D336A13367INSPIRE",
    "a1199a489f_6ABE00F5A1OPENPIPELINE",
    "137f4dfb89_C966B12B4EOPENPIPELINE",
    "551063e3c5_8FCB044F58INSPIRE",
    "37cf2e5706_74D898C7C3OPENPIPELINE",
    "74d7796531_EB81FE6E2BOPENPIPELINE",
    "46b27f92c2_06BA0AF311OPENPIPELINE",
    "32052d9b97_9ABAFDAA93OPENPIPELINE",
]

test = [
    "12fa5e614f_53197F206FOPENPIPELINE",
    "feb7a50f10_JAREDINSPIRE",
    "c2e8370ca3_3340CAC7AEOPENPIPELINE",
    "55ca10d9f1_E8C8441957INSPIRE",
    "5ab849ec40_2F98B8FC82INSPIRE",
    "9254c82db0_9C194DD066OPENPIPELINE",
    "168ac179d9_31328BCCC4OPENPIPELINE",
    "6f93b9026b_F1BFB8B17DOPENPIPELINE",
    "8b0ac1fc28_6688905E16OPENPIPELINE",
    "1553539551_APIGENERATED",
    "7310356a1b_7EAE3AC26AOPENPIPELINE",
    "632de91030_9ABAFDAA93OPENPIPELINE",
    "2f7aabb6e5_0C2B5F6CABOPENPIPELINE",
    "18072ccb69_B2AE5C54EBOPENPIPELINE",
    "8710b98ea0_06E6522D6DINSPIRE",
    "fb74c54103_6ABE00F5A1INSPIRE",
    "25f1c24f30_EB81FE6E2BOPENPIPELINE",
    "39e77bedd0_729FB913CDOPENPIPELINE",
    "e87da4ebdb_29FEA32BC7INSPIRE",
    "546f85625a_39E021DC32INSPIRE",
    "e1d3e6f6ba_B4DE0FB544INSPIRE",
    "eee7d707d4_6DC1FE1DDCOPENPIPELINE",
    "3ff76e84d5_0DD77DFCD7OPENPIPELINE",
    "a0a6f46099_F93BAE5403OPENPIPELINE",
    "420d6b69b8_84B52814D2OPENPIPELINE",
    "d06b2c67d2_2A62B67B52OPENPIPELINE",
    "107f24d6e9_F1BE1D4184INSPIRE",
    "36d5956a21_8F4CE60B77OPENPIPELINE",
    "1726eb08ef_60693DB04DINSPIRE",
    "dabec5e872_E8AD935CEDINSPIRE",
]
class SegDataLoader(Dataset, ABC):
    """ Abstract class serving as a Dataset for 
    pytorch training of interactive models on Gis
    datasets.
    """
    def __init__(self, dataset: str, cfg: Dict[str, Any]):
        """
        Parameters
        ----------
        dataset
            path to dataset
        cfg
            configuration dictionary
        """
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg

    def __len__(self):
        """Defines the length of an epoch"""
        if self.train:
            return self.cfg.EPOCH_SIZE
        return len(self.test_ids)

    @abstractmethod
    def _load_data(self, i):
        return

    @abstractmethod
    def _data_augmentation(self, data):
        pass

    def get_loader(self, batch_size: int, workers: int = 0) -> torch.utils.data.DataLoader:
        """
        Parameters
        ----------
        train_set
            torch dataset
        batch_size
            Batch size
        workers
            Number of sub processes used in the process.

        Returns
        -------
        torch dataloader
        """
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, num_workers=workers, worker_init_fn=self.init_fn
        )

    def split_dataset(self, test_size: float):
        # TODO: Refactoring neaded over here
        dataset_files = glob(os.path.join(self.dataset, "gts/*"))
        if "DroneDeploy" in self.dataset:
            dataset_files = sorted([os.path.basename(i) for i in dataset_files])
            ids = test + val
            test_files = [i + '.tif' for i in ids]
            test_ids, train_ids = [], []
            for i in range(len(dataset_files)):
                if dataset_files[i] in test_files:
                    test_ids.append(i)
                else:
                    train_ids.append(i)
            test_ids = np.asarray(test_ids)
            train_ids = np.asarray(train_ids)
            # test_ids = np.where(test_file == dataset_files)[0]
            # train_ids = np.where(test_file != dataset_files)[0]
        else:
            dataset_ids = np.arange(len(dataset_files))
            if test_size < 1:
                train_ids, test_ids = train_test_split(dataset_ids, test_size=test_size, random_state=42)
                train_ids = train_ids[:int(self.cfg.SUB_TRAIN * len(train_ids))]
            else:
                train_ids, test_ids = dataset_ids, dataset_ids
        if len(train_ids) and len(test_ids):
            return train_ids, test_ids
        dataset_path = os.path.abspath(self.dataset)
        message = "Can't load dataset, propbably path is empty. \n {}".format(dataset_path)
        raise Exception(message)

    @staticmethod
    def init_fn(worker_id):
        """ Initialize numpy seed for torch Dataloader workers."""
        random.seed(np.uint32(torch.initial_seed() + worker_id))
