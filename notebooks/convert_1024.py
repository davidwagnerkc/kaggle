#!/usr/bin/env python
# coding: utf-8
#mkdir ../input/test_1024

#mkdir ../input/train_1024

#mkdir ../input/hpa_1024

import pandas as pd
import numpy as np
from PIL import Image
from multiprocessing import Pool
from pathlib import Path


def convert(z):
    try:
        fn = z.name.replace(z.suffix, '.png')
        if not 'png' in z.suffix or Path(f'../input/hpa_1024/{fn}').exists():
            return
        im = Image.open(z).resize((1024, 1024), Image.LANCZOS)
        im.save(f'../input/hpa_1024/{fn}', format='png')
    except:
        print(z)

#test_dir = Path('../input/test_full_size/')
#train_dir = Path('../input/train_full_size/')
hpa_dir = Path('../input/HPAv18Full/')

p = Pool()

p.map(convert, list(hpa_dir.iterdir()))

