import os
import torch
from run_nerf import train, config_parser

if __name__=='__main__':
    torch.manual_seed(42)
    update_refs = os.getenv("UPDATE_REFS") == '1'

    if update_refs:
        print("Updating refs...")
        basedir = './refs'
    else:
        basedir = '/tmp/refs'
    args = config_parser(['--config', 'configs/lego.txt', '--N_iters', '1', '--i_weights', '1', '--basedir', basedir])
    train(args)

    if not update_refs:
        ref = torch.load('./refs/blender_paper_lego/000001.ckpt')
        new = torch.load('/tmp/refs/blender_paper_lego/000001.ckpt')
        for k in ref:
            print(f"Checking {k}...")
            torch.testing.assert_close(ref[k], new[k])

    print("Done!")
