from pathlib import Path

def delete_global_step_folders(base_path):
    # Convert base_path to a Path object
    base_path = Path(base_path)

    # Check if the base path exists
    if not base_path.exists():
        print(f"Error: The directory '{base_path}' does not exist.")
        return

    # Iterate through all 'checkpoint*/global_step*' directories
    for global_step_dir in base_path.rglob('checkpoint*/global_step*'):
        if global_step_dir.is_dir():
            try:
                # Delete the directory and its contents
                for item in global_step_dir.iterdir():
                    item.unlink() if item.is_file() else item.rmdir()
                global_step_dir.rmdir()  # Remove the now-empty 'global_step*' directory
                print(f"Successfully deleted: {global_step_dir}")
            except Exception as e:
                print(f"Failed to delete {global_step_dir}: {e}")

# Example usage
delete_global_step_folders('./')
