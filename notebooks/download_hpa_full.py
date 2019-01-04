import requests
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
from PIL import Image
from io import BytesIO
df = pd.read_csv('../HPAv18RBGY_wodpl.csv')
def download(id_):
    try:
        hpa_dir = Path('../input/HPAv18Full')
        hpa_dir.mkdir(parents=True, exist_ok=True)
        image_dir, _, image_id = id_.partition('_')
        for c in ['red', 'blue', 'green', 'yellow']:
            url = f'http://v18.proteinatlas.org/images/{image_dir}/{image_id}_{c}.jpg'
            r = requests.get(url)
            image = Image.open(BytesIO(r.content)).resize((1024, 1024), Image.LANCZOS)
            filepath = hpa_dir / f'{id_}_{c}.png'
            image.save(filepath, format='png')
            print(filepath)
        return True
    except:
        print(f'{id_} broke...')
#p = Pool()
#x = p.map(download, df.Id)
#print(all(x))
download('24089_si27_F4_11')
