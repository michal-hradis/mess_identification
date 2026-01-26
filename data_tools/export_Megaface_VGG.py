
import cv2
import argparse
import os
import lmdb

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True, help='Input text file with file in  directories representing identities.')
    parser.add_argument('-o', '--output-name', default='out', help='Name of output file and output lmdb')
    parser.add_argument('-p', '--input-path', default='./', help='Input image path.')
    parser.add_argument('--width', default=128, type=int, help='Width of the final crop.')
    parser.add_argument('--height', default=128, type=int, help='Height of the final crop.')
    parser.add_argument('--show', action='store_true', help='Display stored images.')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    with open(args.input_file, 'r') as f:
        data = [line.strip() for line in f]
    print('Image count', len(data))

    counter = 0
    with lmdb.open(args.output_name, map_size=10000000000) as env:
        with env.begin(write=True) as txn:
            for file_path in data:
                _, idx, file_name = file_path.split('/')
                idx = idx.replace('_', '-')
                img = cv2.imread(os.path.join(args.input_path, file_path))
                if img is None:
                    print('ERROR: unable to read', os.path.join(args.input_path, f'{file_path}'))
                    continue
                img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
                retval, img_data = cv2.imencode('.jpg', img, params=[int(cv2.IMWRITE_JPEG_QUALITY), 98])
                txn.put(key=f"{idx}_{file_name}".encode(), value=img_data.tobytes())
                counter += 1
                if counter % 1000 == 0:
                    print(f'DONE {counter}/{len(data)}')

if __name__ == '__main__':
    main()