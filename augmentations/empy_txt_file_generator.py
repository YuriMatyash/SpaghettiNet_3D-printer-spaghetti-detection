import os

def create_empty_files(folder_path, base_name, count):
    # 1. Create the directory if it doesn't exist
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Directory '{folder_path}' created.")
        except OSError as e:
            print(f"Error creating directory: {e}")
            return

    # 2. Loop through the count to create files
    for i in range(1, count + 1):
        # Construct the filename: e.g., "1_myname.txt"
        filename = f"{i}_{base_name}.txt"
        
        # Join the folder path and filename correctly for the OS
        full_path = os.path.join(folder_path, filename)
        
        try:
            # Open in 'w' (write) mode and close immediately to create an empty file
            with open(full_path, 'w') as f:
                pass 
            print(f"Created: {full_path}")
        except IOError as e:
            print(f"Error creating file {filename}: {e}")

if __name__ == "__main__":
    print("--- File Generator ---")
    
    # Get user inputs
    target_folder = "temp/"
    name_input = "clean"
    count_input = 20

    # Run the function
    create_empty_files(target_folder, name_input, count_input)
    print("\nProcess Complete.")