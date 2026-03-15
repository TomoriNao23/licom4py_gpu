import numpy as np
import sys

def check_npz(filepath):
    print("--- Checking npz file ---")
    data = np.load(filepath)
    a_pt = data["a_pt"]
    
    print(f"a_pt shape: {a_pt.shape}")
    
    # Assuming shape is (6, nx, ny) and (6, nx, ny, 2)
    faces = a_pt.shape[0]
    for i in range(faces):
        print(f"Face {i}:")
        print(f"  a_pt  point(49, 50, :): \n{a_pt[i, 49-3, 50-3, :]}")
        print(f"  a_pt  sum: {a_pt[i].sum():.6g}")
    print("-------------------------\n")

if __name__ == "__main__":
    check_npz("/Users/Chtholly/Documents/2026/code/314/licom4py/field/duogrid_C96.npz")
