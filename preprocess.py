from dataset import Data
from torch.utils.data import DataLoader
from torchvision.utils import save_image


dataset_train = Data(path="/Users/m.rossi/Desktop/research/results/conditioned_3/run9/", mode="train", size=(128,128))

dataloader_train = DataLoader(dataset=dataset_train, batch_size=1)

for batch_idx, (sharp, blur) in enumerate(dataloader_train):
    pass