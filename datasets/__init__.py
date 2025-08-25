from .aircraft_dataset import AircraftDataset
from .bird_dataset import BirdDataset
from .car_dataset import CarDataset
from .ndtwin_dataset import NDTwinDataset, NDTwinVerificationDataset


def get_trainval_datasets(tag, resize):
    if tag == 'aircraft':
        return AircraftDataset(phase='train', resize=resize), AircraftDataset(phase='val', resize=resize)
    elif tag == 'bird':
        return BirdDataset(phase='train', resize=resize), BirdDataset(phase='val', resize=resize)
    elif tag == 'car':
        return CarDataset(phase='train', resize=resize), CarDataset(phase='val', resize=resize)
    elif tag == 'ndtwin':
        return NDTwinDataset(phase='train'), NDTwinVerificationDataset()
    else:
        raise ValueError('Unsupported Tag {}'.format(tag))