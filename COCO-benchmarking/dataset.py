from pycocotools.coco import COCO
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, annotation, data_dir, transforms=False):
        """
        Args:
            annotation: annotation 파일 위치
            data_dir: data가 존재하는 폴더 경로
            transforms : transform 여부

        "images"에 있는 "id"가 "annotations"에 "image_id"와 연결되는 값
        """

        super().__init__()
        self.data_dir = data_dir

        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)
        self.transforms = transforms

        whole_image_ids = self.coco.getImgIds()
        self.image_ids = []

        # to remove not annotated image idx
        self.no_anno_list = []
        for idx in whole_image_ids:
            annotations_ids = self.coco.getAnnIds(imgIds=idx)
            if len(annotations_ids) == 0:
                self.no_anno_list.append(idx)
            else:
                self.image_ids.append(idx)

    def __getitem__(self, index: int):
        # get ground truth annotations
        target_id = self.image_ids[index]
        annotations_ids = self.coco.getAnnIds(imgIds=target_id)  # len() == 5
        annotations = list()

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            print(annotations_ids)

        # parse annotations -> [{'image_id': 397133, 'id': 370509, 'caption': 'A man is in a kitchen making pizzas.'} ...]

        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, ann in enumerate(coco_annotations):
            annotations.append(ann["caption"])

        return target_id, annotations

    def collate_fn(self, batch):
        image_ids, captions = zip(*batch)

        return image_ids, captions

    def __len__(self):
        return len(self.image_ids)
