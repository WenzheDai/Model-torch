

from torch.utils.data import dataset
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.models.yolo.model import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics import cfg
from ultralytics.utils import colorstr


model = DetectionModel(nc=3)

my_cfg = cfg.get_cfg()

img_path = "../dataset/coco128/images/train2017"
batch = 16

yolo_dataset = YOLODataset(
    img_path=img_path,
    imgsz=640,
    batch_size=batch,
    augment=True,  # augmentation
    hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
    rect=False,  # rectangular batches
    cache=None,
    single_cls=False,
    stride=int(32),
    pad=0.0,
    prefix=colorstr("train: "),
    task='detect',
    classes=3,
    data=None,
    fraction=1.0,
)





