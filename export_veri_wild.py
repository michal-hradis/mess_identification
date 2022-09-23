
import cv2
import argparse
import os
import lmdb

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True, help='Input text file with ids.')
    parser.add_argument('-o', '--output-name', default='out', help='Name of output file and output lmdb')
    parser.add_argument('-p', '--input-path', default='./', help='Name of output file and output lmdb')
    parser.add_argument('--width', default=128, type=int, help='Width of the final crop.')
    parser.add_argument('--height', default=128, type=int, help='Height of the final crop.')
    parser.add_argument('--show', action='store_true', help='Display stored images.')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    with open(args.input_file, 'r') as f:
        data = [line.split() for line in f]
    print('IDS count', len(data))

    counter = 0
    with lmdb.open(args.output_name, map_size=10000000000) as env:
        with env.begin(write=True) as txn:
            for file_id, idx, cam_id in data:
                file_id = file_id.split('/')[1]
                img = cv2.imread(os.path.join(args.input_path, f'{file_id}'))
                if img is None:
                    print('ERROR: unable to read', os.path.join(args.input_path, f'{file_id}'))
                    continue
                img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_AREA)
                retval, img_data = cv2.imencode('.jpg', img, params=[int(cv2.IMWRITE_JPEG_QUALITY), 98])
                txn.put(key=f"VehicleID-{idx}_{file_id}".encode(), value=img_data.tobytes())
                counter += 1
                if counter % 1000 == 0:
                    print(f'DONE {counter}/{len(data)}')

if __name__ == '__main__':
    main()