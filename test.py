import pandas as pd
import numpy as np

dates = pd.date_range('1/1/2000', periods=8)

df = pd.DataFrame(np.random.randn(8, 4),
                  index=dates, columns=['A', 'B', 'C', 'D'])

dates_1 = pd.date_range('1/3/2000', periods=2)

print(df)
print(df.loc[dates_1])
