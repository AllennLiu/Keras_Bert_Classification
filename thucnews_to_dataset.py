import os
import pandas as pd
from pathlib import Path
from typing import List, Generator

def data_generator(path: str) -> Generator[List[str], None, None]:
    for r, _, f in os.walk(path):
        label = os.path.basename(r)
        for file in f:
            content = Path(os.path.join(r, file)).read_text(encoding='utf-8')
            yield [ label, content ]

data = data_generator('./THUCNews')
os.makedirs('./data/thucnews', exist_ok=True)
df = pd.DataFrame(columns=[ 'label', 'content' ], data=data)
df_train = df.sample(frac=1).reset_index(drop=True)
df_train.to_csv(os.path.join('./data/thucnews', 'train.csv'), encoding='utf-8', index=False)
# using before 0.125% of train data to be test data
df_test = df_train.head(round(len(df) * .125))
df_test.to_csv(os.path.join('./data/thucnews', 'test.csv'), encoding='utf-8', index=False)
