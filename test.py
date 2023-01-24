import os
import torch
import shutil
import numpy as np
from run_nerf import train
from run_nerf_helpers import load_args

TESTS = {'fern': 'fern_test', 'lego': 'blender_paper_lego'}

if __name__=='__main__':
    update_refs = os.getenv("UPDATE_REFS") == '1'
    test_video = os.getenv("TEST_VIDEO") == '1'

    if update_refs:
        print("Updating refs...")
        basedir = './refs'
    else:
        basedir = '/tmp/refs'
    video_args = ['--i_video', '1'] if test_video else []

    if os.path.exists(basedir):
        shutil.rmtree(basedir)

    for test, expname in TESTS.items():
        print(f"Testing {test}")
        np.random.seed(0)
        torch.manual_seed(42)
        train(load_args(['--config', f'configs/{test}.txt', '--N_iters', '1', '--i_weights', '1', *video_args, '--basedir', basedir]))

        if not update_refs:
            ref = torch.load(f'./refs/{expname}/000001.ckpt')
            new = torch.load(f'/tmp/refs/{expname}/000001.ckpt')
            for k in ref:
                print(f"Checking {k}...")
                torch.testing.assert_close(ref[k], new[k])

            if test_video:
                print("Checking video...")
                ref = np.load(f'./refs/{expname}/{expname}_spiral_000001_rgb.npz')['arr_0']
                new = np.load(f'/tmp/refs/{expname}/{expname}_spiral_000001_rgb.npz')['arr_0']
                np.testing.assert_equal(ref, new)

    print("Done!")
