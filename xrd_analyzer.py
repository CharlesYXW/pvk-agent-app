import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- Configuration ---
DATA_FILE = "experimental_data/xrd/simulated_xrd_data.txt"
OUTPUT_IMAGE = "xrd_spectrum.png"
# --- End Configuration ---

def analyze_xrd_data():
    """
    Loads XRD data, finds peaks, and generates a plot.
    """
    print(f"Loading XRD data from '{DATA_FILE}'...")
    try:
        data = np.loadtxt(DATA_FILE, comments="#", delimiter=",")
        angle = data[:, 0]
        intensity = data[:, 1]
    except Exception as e:
        print(f"Error: Could not load or parse the data file. {e}")
        return

    print("Data loaded successfully. Finding peaks...")
    peaks, _ = find_peaks(intensity, height=600, distance=10)

    if peaks.size == 0:
        print("No significant peaks found.")
    else:
        print(f"Found {len(peaks)} major peaks at 2-theta angles:")
        for peak_index in peaks:
            print(f"- {angle[peak_index]:.2f}°")

    # --- Plotting ---
    print(f"Generating plot and saving to '{OUTPUT_IMAGE}'...")
    plt.figure(figsize=(12, 7))
    plt.plot(angle, intensity, label="XRD Spectrum")
    plt.plot(angle[peaks], intensity[peaks], "x", markersize=10, label="Detected Peaks")

    for peak_index in peaks:
        plt.annotate(f"{angle[peak_index]:.2f}°",
                     (angle[peak_index], intensity[peak_index]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=9)

    plt.title("Simulated XRD Spectrum Analysis")
    plt.xlabel("2-Theta Angle (°)")
    plt.ylabel("Intensity (Arbitrary Units)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(OUTPUT_IMAGE)
    print("Plot saved successfully.")

if __name__ == "__main__":
    analyze_xrd_data()
