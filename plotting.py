import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('april 2 and 3 example.xlsx', index_col=0, parse_dates=True)

print(df)