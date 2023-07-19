import os
import glob
import shutil
import time

subj = 1

while subj < 9:
    import pdb; pdb.set_trace()
    ckpt_path = glob.glob("/SSD/guest/qasymjomart/algonauts/breil-algonauts-challenge/vith/epoch=5*")
    print(ckpt_path)
    if  len(ckpt_path) > 0:
        shutil.copy(ckpt_path[0], f"/SSD/guest/qasymjomart/algonauts/breil-algonauts-challenge/vith/ckpts/epoch=20_subj{subj}")
        print(f'{subj} checkpoint at 20th epoch is copied to ckpts.')
        subj += 1
        time.sleep(1000)
        continue
    else:
        continue
