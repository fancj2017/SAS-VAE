import os, time
import numpy as np
import torch
import MinkowskiEngine as ME
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from data_utils import array2vector, istopk, sort_spare_tensor, load_sparse_tensor, scale_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo,xn

from gpcc import gpcc_encode, gpcc_decode
from pc_error import pc_error

from pcc_model import PCCModel

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--filedir", default='longdress_vox10_1300.ply')
    parser.add_argument("--res", type=int, default=1024, help='resolution')

    args = parser.parse_args()
    filedir = args.filedir
    filename = os.path.split(filedir)[-1].split('.')[0]
    # static pc error

    #pc_error_metrics = pc_error(filename+'.ply', filename+'_dec.ply', res=args.res, normal=True, show=False)
    pc_error_metrics = pc_error('/home/cj/桌面/PCGCv2-master/longdress_vox10_1300.ply', '/home/cj/桌面/PCGCv2-master/output/longdress_vox10_1300_r7_dec.ply', res=args.res, normal=True, show=False)

    print('pc_error_metrics:', pc_error_metrics)
    print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
    print('D2 PSNR:\t', pc_error_metrics["mseF,PSNR (p2plane)"][0])
