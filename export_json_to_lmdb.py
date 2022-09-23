import random

import cv2
import numpy as np
import argparse
import json
import os
import lmdb
from collections import defaultdict
from crop import RandomGeometricGenerator_2


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', required=True, help='Input json file with positions.')
    parser.add_argument('-o', '--output-name', default='out', help='Name of output file and output lmdb')
    parser.add_argument('-p', '--input-path', default='./', help='Name of output file and output lmdb')
    parser.add_argument('--width', default=128, type=int, help='Width of the final crop.')
    parser.add_argument('--height', default=128, type=int, help='Height of the final crop.')
    parser.add_argument('--intersection-area', default=0.2, type=float, help='The relative size of the object.')
    parser.add_argument('--offset-y', default=0.3, type=float, help='If positive, the object is lower. ')
    parser.add_argument('--show', action='store_true', help='Display stored images.')
    parser.add_argument('--test-count', default=1000, type=int, help='Number of test identities.')

    args = parser.parse_args()
    return args


def parse_np_points_from_string(string):
    points = [x.split(':') for x in string.split(',')]
    try:
        points = np.asarray([[float(x[0]), float(x[1])] for x in points])
    except ValueError:
        return np.zeros((0, 2))
    return points


def export(db_path, input_path, data, lp_names, cropper, width, height, show):
    exit = False
    count = 0
    with lmdb.open(db_path, map_size=10000000000) as env:
        with env.begin(write=True) as txn:
            counts = defaultdict(int)

            # Add data and key value
            for lp_text in lp_names:

                if exit:
                    break

                for lp in data[lp_text]:
                    img = cv2.imread(os.path.join(input_path, lp['name']))
                    if img is None:
                        print('ERROR: unable to read', os.path.join(input_path, lp['name']))
                        continue

                    points = lp['points']
                    points = parse_np_points_from_string(points)
                    img, points, T = cropper.generate_crop(img, points, ar=lp['ratio'])
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                    retval, img_data = cv2.imencode('.jpg', img, params=[int(cv2.IMWRITE_JPEG_QUALITY), 98])
                    txn.put(key=f"{lp_text}_{counts[lp_text]}".encode(), value=img_data.tobytes())
                    counts[lp_text] += 1

                    if show:
                        cv2.imshow('i', img)
                        k = cv2.waitKey()
                        if k == 27:
                            exit = True
                            break

                count += 1
                if count % 100 == 0:
                    print(f'DONE {count}/{len(lp_names)}')


def main():
    args = parse_arguments()

    cropper = RandomGeometricGenerator_2('c', resolution=(args.width*2, args.height*2), points_2=False,
                                         intersection_area=args.intersection_area, offset_y=args.offset_y,
                                         absolute_crop=False, aligned_bbox=True)

    with open(args.json, 'r') as f:
        data = json.load(f)
    print('IDS count', len(data.keys()))

    lp_codes = [d for d in data if len(d) >= 5]
    random.shuffle(lp_codes)
    tst_lp_codes = lp_codes[:args.test_count]
    trn_lp_codes = lp_codes[args.test_count:]

    print(f'TRAIN/TEST: {len(trn_lp_codes)}  {len(tst_lp_codes)}')
    if tst_lp_codes:
        export(args.output_name + '.tst', args.input_path, data, tst_lp_codes, cropper, args.width, args.height, args.show)

    if trn_lp_codes:
        export(args.output_name + '.trn', args.input_path, data, trn_lp_codes, cropper, args.width, args.height, args.show)


if __name__ == '__main__':
    main()