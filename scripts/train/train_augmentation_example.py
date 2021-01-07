import os
import argparse
from sys import platform


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,  required=True)
parser.add_argument('--cluster_id', type=int,  required=True)
parser.add_argument('--gpu_id', type=int,  default=0)
args,unknown = parser.parse_known_args()

if len(unknown) != 0:
    print(unknown)
    exit(-1)


if platform == "linux" or platform == "linux2":
    prefix = " CUDA_VISIBLE_DEVICES=%d " % args.gpu_id
elif platform == "win32":
    prefix = " set CUDA_VISIBLE_DEVICES=%d &"  % args.gpu_id


template = prefix + ' python ../../src/train_neural_render_aug.py --dataDir ../../data/%s/Train/PL/Cluster_%d  --logDir ../../log_results/%s/Aug/log_cluster_%d    --max_steps 200000  --texture_channels 30 --lr 0.0002    --data_max_val %f --keep_max_val %f --rescale_input 1.0 --rescale_output 1.0  --augDataDir ../../data/%s/Train/Env/Cluster_%d --augRescale 0.32 --checkpoint ../../log_results/%s/PL/log_cluster_%d '


configs = {
    'tree': [2.0, 2.0],
    'sphere': [10.0, 10.0],
    'pig': [3.834, 1.0],
    'pixiu': [1.0, 1.0],
    'fur': [1.0, 1.0]
}


if __name__ == '__main__':
    name = args.name
    cluster_id = args.cluster_id
    data_max_val, keep_max_val = configs['tree']

    cmd = template % (name, cluster_id, name, cluster_id, data_max_val, keep_max_val, name, cluster_id, name, cluster_id)

    print(cmd)
    os.system(cmd)


