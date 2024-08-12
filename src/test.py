import pandas as pd
import numpy as np
import torch.nn as nn

foo = np.array([1,2,4])

print(foo)

bar = pd.DataFrame(foo)

class myNet():
    def __init__(self, input, output) -> None:
        fc1 = nn.Linear(input, output)


print("bar: ", bar)
