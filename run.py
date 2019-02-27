import sys
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

values = pd.read_csv(sys.argv[1], header=None).values
plt.plot([x[0] for x in values], [y[1] for y in values], 'rx')
plt.show()
