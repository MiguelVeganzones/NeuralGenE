from multiprocessing.connection import wait
import sys
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import pyodbc

if __name__ == "__main__":
    cnxn_str = ("Driver={ODBC Driver 17 for SQL Server};"
                "Server=(localdb)\MSSQLLocalDB;"
                "Database=Training_data;"
                #"UID=data_login;"
                #"PWD=1234;"
                #"Trusted_connection=yes;"
                )
    cnxn = pyodbc.connect(cnxn_str)
    
    exit()
    print(sys.argv)
    if len(sys.argv) != 3:
        exit()

    path = sys.argv[1]
    length = int(sys.argv[2])
    print(path)
    print(length)

    prev_data_idx = -1
    data_idx = -1

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_ylim(-1.05, 1.05)
    x = [i for i in range(length)]
    plot_line, = ax.plot(x, x)

    with open(path) as f:
        while True:
            f.seek(0)
            first_val = f.readline()
            #print(first_val)

            if first_val is None or not first_val or first_val in ('', ' '):
                sleep(1)
                continue

            data_idx = int(first_val)
            if data_idx < prev_data_idx:
                #print(data_idx, prev_data_idx)
                continue 
            else:
                prev_data_idx = data_idx
                
            sleep(0.01)

            f.seek(1)
            for line in f: # read rest of lines
                plot_line.set_xdata(x)
                data = [float(x) for x in line.split()]
                plot_line.set_ydata(data)

            fig.canvas.draw()
            fig.canvas.flush_events()

    print("PyPlotting terminated")

