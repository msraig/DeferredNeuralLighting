"""
Example:

Download the training dataset of tree:
$ python .\download_dataset.py --scene tree --mode train

Download subset of training dataset of tree:
$ python .\download_dataset.py --scene tree --mode train --cluster_indices 0,3,5

Download testing dataset of tree:
$ python .\download_dataset.py --scene tree --mode test

"""

import requests
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str,  required=True, choices=['tree', 'pig', 'sphere', 'pixiu', 'furscene'])
parser.add_argument('--output_path', type=str, default=None)

parser.add_argument('--mode', type=str,  required=True, choices=['test', 'train'])
parser.add_argument('--cluster_indices', type=str,  default=None)

args,unknown = parser.parse_known_args()

if args.output_path is None:
    cur_path = os.path.dirname(os.path.abspath(__file__))
    args.output_path = os.path.join(cur_path, 'data', args.scene)
print("Download into %s" % args.output_path)


if args.mode == 'train':
    if args.cluster_indices is None:
        args.cluster_indices = list(range(13))
    else:
        args.cluster_indices = [int(i) for i in args.cluster_indices.split(',')]
    print("Cluster indices: %s" % (','.join([str(i) for i in args.cluster_indices])))

def download_file(url, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    filename = url.split('/')[-1]
    dst_name = os.path.join(dst_path, filename)
    with requests.get(url, stream=True) as r:
        total_length = int(r.headers.get('content-length'))
        print("Size: %.2f GB. Url: %s, DstFile: %s" % (int(float(total_length) / 1024.0 / 1024.0 / 1024.0), url, dst_name) ) 
        r.raise_for_status()
        with open(dst_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
                f.flush()
    return dst_name

if __name__ == '__main__':
    if args.mode == 'test':
        test_url_template = 'https://igpublicshare.z20.web.core.windows.net/DeferredNeuralLighting/data/%s/Test.zip'
        url = test_url_template % args.scene
        download_file(url, args.output_path)

    elif args.mode == 'train':
        args.output_path = os.path.join(args.output_path, 'Train/PL')
        if args.scene in ['tree', 'pig', 'sphere']:
            train_url_synthetic_template = 'https://igpublicshare.z20.web.core.windows.net/DeferredNeuralLighting/data/%s/Train/PL/Cluster_%d/Cluster_%d_%d.tfrecords'
            
            for cluster_idx in args.cluster_indices:
                download_file(train_url_synthetic_template % (args.scene, cluster_idx, cluster_idx, 0), os.path.join(args.output_path, 'Cluster_%d' % cluster_idx))
                if cluster_idx  == 3:
                    download_file(train_url_synthetic_template % (args.scene, cluster_idx, cluster_idx, 1), os.path.join(args.output_path, 'Cluster_%d' % cluster_idx))
        else:
            if args.scene == 'furscene':
                train_url_real_template = 'https://igpublicshare.z20.web.core.windows.net/DeferredNeuralLighting/data/%s/Train/PL/Cluster_%d/Cluster_%d_0seq_%d.tfrecords'
            else:
                train_url_real_template = 'https://igpublicshare.z20.web.core.windows.net/DeferredNeuralLighting/data/%s/Train/PL/Cluster_%d/Cluster_%d_0seq%d.tfrecords'

            parts = 3 if args.scene == 'pixiu' else 6

            for cluster_idx in args.cluster_indices:
                for part_idx in range(1, parts + 1):
                    if args.scene == 'pixiu' and cluster_idx == 3 and part_idx == 1:
                        continue
                    download_file(train_url_real_template % (args.scene, cluster_idx, cluster_idx, part_idx), os.path.join(args.output_path, 'Cluster_%d' % cluster_idx))

