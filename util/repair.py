import json

ann_path = "F:/coco/VOC2028/annotations/instances_train2017.json"

with open(ann_path, "r", encoding="utf-8") as f:
    data = json.load(f)

count = 0
for img in data["images"]:
    if img["file_name"].endswith(".jpeg.jpg"):
        img["file_name"] = img["file_name"].replace(".jpeg.jpg", ".jpg")
        count += 1

with open(ann_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print(f"修复完成！共修改 {count} 个文件名。")
