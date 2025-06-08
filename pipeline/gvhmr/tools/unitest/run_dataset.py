import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

def get_dataset(DATA_TYPE):
    
    if DATA_TYPE == "BEDLAM1":
        from hmr4d.dataset.bedlam.bedlam1_prhmr import Bedlam1PrhmrDataset
        return Bedlam1PrhmrDataset(version="hand_v2")
    
    if DATA_TYPE == "BEDLAM2":
        from hmr4d.dataset.bedlam.bedlam2_prhmr import Bedlam2PrhmrDataset
        return Bedlam2PrhmrDataset(version="hand_v2")
    
    if DATA_TYPE == "BEDLAM_V2":
        from hmr4d.dataset.bedlam.bedlam import BedlamDatasetV2

        return BedlamDatasetV2()

    if DATA_TYPE == "3DPW_TRAIN":
        from hmr4d.dataset.threedpw.threedpw_motion_train import ThreedpwSmplDataset

        return ThreedpwSmplDataset()
    
    if DATA_TYPE == "EMDB_1":
        from hmr4d.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
        return EmdbSmplFullSeqDataset(split=1, flip_test=False, version="hand_v2")
    
    if DATA_TYPE == "EMDB_2":
        from hmr4d.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
        return EmdbSmplFullSeqDataset(split=2, flip_test=False, version="hand_v2")
    
    if DATA_TYPE == "RICH":
        from hmr4d.dataset.rich.rich_motion_test import RichSmplFullSeqDataset
        return RichSmplFullSeqDataset()


if __name__ == "__main__":
    DATA_TYPE = sys.argv[1]
    dataset = get_dataset(DATA_TYPE)
    print(len(dataset))

    data = dataset[0]

    from hmr4d.datamodule.mocap_trainX_testY import collate_fn

    loader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        batch_size=1,
        collate_fn=collate_fn,
    )
    i = 0
    for batch in tqdm(loader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
        i += 1
        # if i == 20:
        #     raise AssertionError
        # time.sleep(0.2)
        if i > 40:
            break
    print('Done!')
