import json
import os
from shutil import copy2
from math import ceil

# 数据集路径
datasets = [
    "F:/coco",
    "F:/human_head_coco",
]

output_dir = "F:/helmet3"

# 创建输出目录
for d in ["train2017", "val2017", "annotations"]:
    os.makedirs(os.path.join(output_dir, d), exist_ok=True)

# 统一类别
categories = [
    {"supercategory": "none", "id": 1, "name": "helmet"},
    {"supercategory": "none", "id": 2, "name": "head"},
    {"supercategory": "none", "id": 3, "name": "person"}
]

def merge_coco(datasets, split):
    merged = {"images": [], "annotations": [], "categories": categories}
    img_id_counter = 1
    ann_id_counter = 1
    img_name_set = set()

    for dataset in datasets:
        ann_file = os.path.join(dataset, f"annotations/instances_{split}2017.json")
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        imgs = data["images"]
        anns = data["annotations"]

        # human_head_coco 取一半
        if "human_head_coco" in dataset:
            half = ceil(len(imgs) / 4)
            imgs = imgs[:half]
            valid_ids = set(img["id"] for img in imgs)
            anns = [ann for ann in anns if ann["image_id"] in valid_ids]

        print(f"{dataset} ({split}) 处理 {len(imgs)} 张图片，{len(anns)} 个标注")

        old_to_new_id = {}
        # 处理图片
        for img in imgs:
            old_id = img["id"]
            new_id = img_id_counter
            img_id_counter += 1
            old_to_new_id[old_id] = new_id
            img["id"] = new_id

            # 避免图片重名
            orig_name = img["file_name"]
            new_name = f"{os.path.basename(dataset)}_{orig_name}" if orig_name in img_name_set else orig_name
            img_name_set.add(new_name)
            img["file_name"] = new_name

            # 复制图片
            src_path = os.path.join(dataset, split + "2017", orig_name)
            dst_path = os.path.join(output_dir, split + "2017", new_name)
            if os.path.exists(src_path):
                copy2(src_path, dst_path)
            else:
                print(f"⚠️ 图片不存在: {src_path}")

            merged["images"].append(img)

        # 处理标注
        for ann in anns:
            ann["id"] = ann_id_counter
            ann_id_counter += 1
            ann["image_id"] = old_to_new_id[ann["image_id"]]
            merged["annotations"].append(ann)

    # 保存合并后的 JSON
    out_file = os.path.join(output_dir, f"annotations/instances_{split}2017.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)

    print(f"{split} 合并完成，共 {len(merged['images'])} 张图片，{len(merged['annotations'])} 个标注\n")


# 合并 train 和 val
merge_coco(datasets, "train")
merge_coco(datasets, "val")
