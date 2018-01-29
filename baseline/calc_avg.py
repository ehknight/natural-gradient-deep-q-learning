import numpy as np
import matplotlib.pyplot as plt

def bestavg(arr, max_len, n=100):
    # assert len(arr) > max_len
    avgs = [np.average(arr[i:i+n]) for i in range(min(max_len, len(arr))-n)]
    # avgs cut off last value, because baselines always return '0' as last element of rewards
    return max(avgs), np.argmax(np.asarray(avgs))

def avgs(arr, max_len, n=100):
    avgs = [np.average(arr[i:i+n]) for i in range(min(max_len, len(arr))-n)]
    return avgs


calc_avg = bestavg
avg = bestavg
