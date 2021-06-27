from torchvision.datasets import CIFAR10
import torchvision.transforms as T

from torch.utils.data import DataLoader


mean_val, std_val = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def get_loader(root, train=True, batch_size=32, num_worker=8, ro=30, tr=64, vr=64, drop_last=True):
    train_transforms = T.Compose([
        T.RandomRotation(ro),
        T.RandomResizedCrop(tr),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean_val,std_val)
    ])

    valid_transforms= T.Compose([
        T.Resize(vr),
        T.ToTensor(),
        T.Normalize(mean_val,std_val)
    ])
    if train:
        dset = CIFAR10(root=root, train=True, download=True, transform=train_transforms)
        loader = DataLoader(dset, batch_size=batch_size, num_workers=num_worker, drop_last=drop_last)
        print('trainset successfully loaded')
    else:
        dset = CIFAR10(root=root, train=False, download=True, transform=valid_transforms)
        loader = DataLoader(dset, batch_size=batch_size, num_workers=num_worker, drop_last=drop_last)
        print('validset successfully loaded')
    return loader, dset
        
def train_loader(root,  batch_size=32, num_worker=8, ro=30, tr=64, drop_last=True):
    return get_loader(root=root, train=True, 
                      batch_size=batch_size, num_worker=num_worker, ro=ro,
                      tr=tr, drop_last=drop_last)

def valid_loader(root, batch_size=32, num_worker=8, vr=64, drop_last=True):
    return get_loader(root=root, train=False, 
                      batch_size=batch_size, num_worker=num_worker, 
                      vr=vr, drop_last=drop_last)
