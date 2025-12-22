#!/usr/bin/env python3
import xml.etree.ElementTree as ET, json, os, glob

def voc2coco(voc_root, out_json):
    jpg_dir  = os.path.join(voc_root, 'JPEGImages')
    xml_dir  = os.path.join(voc_root, 'Annotations')
    assert os.path.isdir(jpg_dir) and os.path.isdir(xml_dir)

    # 1. 扫描全部类别
    cats = sorted({ET.parse(f).findtext('.//name') for f in glob.glob(xml_dir+'/*.xml')})
    cat2id = {n: i+1 for i, n in enumerate(cats)}

    coco = {'images': [], 'annotations': [], 'categories': [{'id': i, 'name': n} for n, i in cat2id.items()]}
    img_id = ann_id = 1

    for xml_path in glob.glob(xml_dir+'/*.xml'):
        root = ET.parse(xml_path).getroot()
        fname = root.findtext('filename')
        if not fname.lower().endswith(('.jpg', '.png')):      # 补扩展名
            fname += '.jpg'
        w = int(root.findtext('.//width'))
        h = int(root.findtext('.//height'))
        coco['images'].append({'id': img_id, 'file_name': fname, 'width': w, 'height': h})

        for obj in root.iter('object'):
            cat = obj.findtext('name')
            if cat not in cat2id: continue
            b = obj.find('bndbox')
            x1, y1, x2, y2 = [float(b.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
            coco['annotations'].append({
                'id': ann_id, 'image_id': img_id, 'category_id': cat2id[cat],
                'bbox': [x1, y1, x2-x1, y2-y1], 'area': (x2-x1)*(y2-y1), 'iscrowd': 0
            })
            ann_id += 1
        img_id += 1

    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    json.dump(coco, open(out_json, 'w'), indent=2)
    print('Convert done:', len(coco['images']), 'images', len(coco['annotations']), 'boxes')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_root', required=True, help='VOC 根目录，含 JPEGImages/ Annotations/')
    parser.add_argument('--out_json', required=True, help='输出 coco json')
    args = parser.parse_args()
    voc2coco(args.voc_root, args.out_json)