import lmdb
import argparse
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Copy LMDB')
    parser.add_argument('--src', type=str, required=True, help='Source LMDB')
    parser.add_argument('--dst', type=str, required=True, help='Destination LMDB')
    parser.add_argument('--remove-prefix', default=[], nargs='+', help='Do not copy keys with these prefixes')
    parser.add_argument('--keep-prefix', default=[], nargs='+', help='Only copy keys with these prefixes')
    return parser.parse_args()


def copy_lmdb(src, dst, remove_prefix: list = [], keep_prefix: list = []):
    env_src = lmdb.open(src, readonly=True, lock=False, readahead=True, meminit=False, create=False)
    env_dst = lmdb.open(dst, map_size=1024 ** 4)
    txn_src = env_src.begin()
    txn_dst = env_dst.begin(write=True)

    copy_count = 0

    for key, value in tqdm(txn_src.cursor()):
        key_str = key.decode()
        if any([key_str.startswith(prefix) for prefix in remove_prefix]):
            continue
        if keep_prefix and not any([key_str.startswith(prefix) for prefix in keep_prefix]):
            continue
        txn_dst.put(key, value)
        copy_count += 1

    txn_dst.commit()
    env_src.close()
    env_dst.close()

    logging.info(f'Copied {copy_count} keys from {src} to {dst}')


def main():
    args = parse_args()
    copy_lmdb(args.src, args.dst, args.remove_prefix, args.keep_prefix)


if __name__ == '__main__':
    main()


