import numpy as np
import matplotlib.pyplot as plt
import os

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(path_XYs, output_path):
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define your list of colors here
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(path_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
   
    plt.savefig(output_path, format='jpg')
    plt.close(fig)  # Close the figure to free up memory

def main():
    # Define the CSV file path
    csv_path = "/Users/srivatsapalepu/Downloads/problems/occlusion1.csv"
    path_XYs = read_csv(csv_path)
    
    base_name, _ = os.path.splitext(csv_path)
    output_path = f"{base_name}.jpg"
    
    plot(path_XYs, output_path)
    print(f"Plot saved as {output_path}")

if __name__ == "__main__":
    main()