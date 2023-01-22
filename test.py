import os
import torch
import shutil
import numpy as np
from run_nerf import train, config_parser

TESTS = {'fern': 'fern_test', 'lego': 'blender_paper_lego'}

if __name__=='__main__':
    update_refs = os.getenv("UPDATE_REFS") == '1'

    if update_refs:
        print("Updating refs...")
        basedir = './refs'
    else:
        basedir = '/tmp/refs'

    if os.path.exists(basedir):
        shutil.rmtree(basedir)

    for test, expname in TESTS.items():
        print(f"Testing {test}")
        np.random.seed(0)
        torch.manual_seed(42)
        args = config_parser(['--config', f'configs/{test}.txt', '--N_iters', '1', '--i_weights', '1', '--basedir', basedir])
        train(args)

        if not update_refs:
            ref = torch.load(f'./refs/{expname}/000001.ckpt')
            new = torch.load(f'/tmp/refs/{expname}/000001.ckpt')
            for k in ref:
                print(f"Checking {k}...")
                torch.testing.assert_close(ref[k], new[k])

    print("Done!")
