import argparse
import os
import lmdb
import glob
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', required=True, help='Input directory with images.')
    parser.add_argument('-o', '--output-path', default='out', help='Name of output file and output lmdb')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    with lmdb.open(args.output_path, map_size=10000000000) as env:
        with env.begin(write=True) as txn:

            for file_name in tqdm(glob.glob(os.path.join(args.input_path, '*.jpg'))):
                # just put the image in the lmdb as is (just a binary copy)
                with open(file_name, 'rb') as f:
                    image_data = f.read()
                txn.put(key=f"{os.path.basename(file_name)}".encode(), value=image_data)


if __name__ == '__main__':
    main()