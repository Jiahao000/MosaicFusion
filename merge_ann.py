import argparse
import copy
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge MosaicFusion annotations into LVIS annotations')

    parser.add_argument(
        '--lvis_path',
        type=str,
        default='data/lvis/meta/lvis_v1_train.json',
        help='LVIS json file path')

    parser.add_argument(
        '--mosaic_path',
        type=str,
        default='output/annotations/lvis_v1_train_mosaicfusion.json',
        help='MosaicFusion json file path')

    parser.add_argument(
        '--save_path',
        type=str,
        default='output/annotations/lvis_v1_train_merged.json',
        help='merged json file path')

    args = parser.parse_args()
    return args


def main():
    # init
    args = parse_args()
    lvis_data = json.load(open(args.lvis_path))
    print(
        f"LVIS: {len(lvis_data['images'])} images, "
        f"{len(lvis_data['annotations'])} annotations, "
        f"{len(lvis_data['categories'])} categories")
    sd_data = json.load(open(args.sd_path))
    print(
        f"MosaicFusion: {len(sd_data['images'])} images, "
        f"{len(sd_data['annotations'])} annotations, "
        f"{len(sd_data['categories'])} categories")
    merged_data = copy.deepcopy(lvis_data)
    for a in sd_data['annotations']:
        a['id'] += 1270141  # lvis has 1270141 instance annotations, new id starts from 1270142
    for c in sd_data['categories']:
        assert merged_data['categories'][c['id'] - 1]['id'] == c['id']
        merged_data['categories'][c['id'] - 1]['image_count'] += c['image_count']
        merged_data['categories'][c['id'] - 1]['instance_count'] += c['instance_count']
    merged_data['images'].extend(sd_data['images'])
    merged_data['annotations'].extend(sd_data['annotations'])
    assert lvis_data['images'] + sd_data['images'] == merged_data['images']
    assert lvis_data['annotations'] + sd_data['annotations'] == merged_data['annotations']
    assert len(lvis_data['categories']) == len(merged_data['categories']) == 1203
    json.dump(merged_data, open(args.save_path, 'w'))
    print(
        f"Done: {len(merged_data['images'])} images, "
        f"{len(merged_data['annotations'])} annotations, "
        f"{len(merged_data['categories'])} categories")


if __name__ == '__main__':
    main()
