from src.op import Operator
from src.constants import *
#import matplotlib.pyplot as plot
import numpy as np
from src.constants import *
import time

def main():
    mu, sigma = 10, 3
    start = time.time()
    #arrival_rates = np.random.poisson(lam=arrival_rates,size=(1, RUNTIME))
    #[np.random.rand() for _ in range(1000*60*60*24)]
    s = np.random.rand(1500*60*60*24)
    n = np.random.normal(mu, sigma, 1500) * 8
    end = time.time()
    print(s)
    print("Time elasped: ", end-start)

if __name__ == '__main__':
    main()