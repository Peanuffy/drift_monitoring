from sklearn.datasets import fetch_openml
import pandas as pd

data = fetch_openml("diamonds", version=1, as_frame=True)
df = data.frame
df.to_csv("data/diamonds.csv", index=False)
