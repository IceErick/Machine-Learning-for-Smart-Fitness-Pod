import os
import pandas as pd
import numpy as np
from data_preprocessing import create_sliding_windows
# Import the create_sliding_windows function defined previously

# Define action keywords and their corresponding folder/filename mappings
# Format: 'keyword': Label_ID
LABEL_MAP = {
    'rest': 0,      # Files containing 'rest' in name → label 0 (walking, drinking, random movement all named rest_xxx.csv)
    'squat': 1,     # Files containing 'squat' in name → label 1
    'bicep': 2,     # Files containing 'bicep' in name → label 2 (bicep curl)
    'bench': 3,
    'run': 4,
}

def load_and_process_all_files(data_dir):
    all_X = []
    all_y = []

    # Iterate over all CSV files in the data directory
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue

        file_path = os.path.join(data_dir, filename)

        # Auto-match label
        current_label = -1
        for keyword, label_id in LABEL_MAP.items():
            if keyword in filename.lower(): # Case-insensitive
                current_label = label_id
                break

        if current_label == -1:
            print(f"Skipping file (no matching label): {filename}")
            continue

        print(f"Processing: {filename} -> Label {current_label}")

        try:
            df = pd.read_csv(file_path)
            # Call the sliding window function
            X_chunk, y_chunk = create_sliding_windows(df, current_label)

            if len(X_chunk) > 0:
                all_X.append(X_chunk)
                all_y.append(y_chunk)
                print(f"  - Generated samples: {len(X_chunk)}")
            else:
                print(f"  - Data too short, no samples generated")

        except Exception as e:
            print(f"  - Processing error: {e}")

    # Merge all data
    if len(all_X) > 0:
        X_final = np.concatenate(all_X, axis=0)
        y_final = np.concatenate(all_y, axis=0)
        return X_final, y_final
    else:
        return np.array([]), np.array([])

# === Main Program ===
if __name__ == "__main__":
    DATA_DIR = 'data/raw2'

    X_train, y_train = load_and_process_all_files(DATA_DIR)

    if len(X_train) > 0:
        print("=" * 30)
        print(f"Processing completed!")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        # Quick check on class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution:", dict(zip(unique, counts)))

        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
        print(".npy files saved")
    else:
        print("Error: No data was generated. Please check the folder path and filenames.")
