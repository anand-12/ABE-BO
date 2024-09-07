import os
import shutil

def copy_and_rename_files(base_path='.'):
    # Iterate through all subdirectories in the base path
    for function_dir in os.listdir(base_path):
        function_path = os.path.join(base_path, function_dir)
        
        # Check if it's a directory
        if os.path.isdir(function_path):
            # Iterate through files in the function directory
            for filename in os.listdir(function_path):
                # Check if the file starts with "GPHedge" and ends with ".npy"
                if filename.startswith("GPHedge") and filename.endswith(".npy"):
                    # Construct the new filename
                    new_filename = f"{function_dir}_{filename}"
                    
                    # Construct the full paths
                    src_path = os.path.join(function_path, filename)
                    dst_path = os.path.join(base_path, new_filename)
                    
                    # Copy and rename the file
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied and renamed: {src_path} -> {dst_path}")

if __name__ == "__main__":
    copy_and_rename_files()
    print("File copying and renaming completed.")
