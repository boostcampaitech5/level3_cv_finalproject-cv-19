import json
import os

from PIL import Image
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


class ImageNetDataset(Dataset):
    def __init__(self, annotation, data_dir, transforms=False):
        """
        Args:
            annotation: annotation 파일 위치
            data_dir: data가 존재하는 폴더 경로
            transforms : transform 여부

        "wnid"가 folder name, "synset"이 class name
        """

        super().__init__()
        folders = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]

        self.images = []
        for folder in folders:
            tmp_lst = os.listdir(folder)
            if len(tmp_lst) != 50:
                print(len(tmp_lst))
            for img in tmp_lst:
                self.images.append(os.path.join(folder, img))

        with open(annotation) as f:
            self.label = json.load(f)

        self.classes = dict()
        for i in self.label:
            self.classes.update({i["wnid"]: i["cid"]})

        self.transforms = transforms

    def __getitem__(self, index: int):
        # get ground truth annotations
        img_path = self.images[index]
        img = Image.open(self.images[index])
        if self.transforms:
            img = self.transforms(img)

        class_id = self.classes[img_path.split("/")[-2]]
        class_label = self.label[class_id]["cid"]

        return img, class_label

    def collate_fn(self, batch):
        image_ids, captions = zip(*batch)

        return image_ids, captions

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    import torch

    import clip

    model, preprocess = clip.load("ViT-B/32", device="cuda")

    my_dataset = ImageNetDataset(annotation="/opt/ml/ImageNet/class_info.json", data_dir="/opt/ml/ImageNet/val", transforms=preprocess)
    dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=2, shuffle=False, drop_last=False)

    for idx, (img, cls) in enumerate(dataloader):
        print(cls)
        if idx % 5 == 0:
            break

    # img, cls = next(iter(dataloader))
    # img = torch.tensor(np.stack(img))  # np.stack(img)
    # print(img.shape)
    # print(cls)
