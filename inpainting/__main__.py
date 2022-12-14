import os 
import sys
import subprocess

import time
from datetime import datetime

if __name__ == "__main__":
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d__%H-%M-%S")
    print("Starting __main__ script at:" +  current_time)

    print("************ Running Command *********************")
    start = time.time()
    outputMode = subprocess.call(['python','inpainting/WGAIN_CELEBA.py','2>&1', '|', 'tee', 'train_CELEBA.log'])
    if outputMode != 0:
        raise Exception("Command failed to train!")
    timeTakenMin = (time.time()-start)/60
    print("Successfully Done training "+str(timeTakenMin)+"min")
