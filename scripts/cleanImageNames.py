import os
import re
import sys

def clean_file_names(directory, max_length=255):
    """
    Rename files in the specified directory to remove special characters,
    handle name conflicts, truncate names that exceed the maximum length, 
    and optionally shorten overly long names.

    Args:
        directory (str): Path to the directory containing the files to rename.
        max_length (int): Maximum allowed length for filenames (default is 255).
    """
    def add_long_path_prefix(path):
        """Add long path prefix for Windows systems."""
        if sys.platform == 'win32' and not path.startswith('\\\\?\\'):
            return f"\\\\?\\{os.path.abspath(path)}"
        return path

    for root, _, files in os.walk(directory):
        for file in files:
            # Original file path
            original_path = os.path.join(root, file)

            # Normalize path separators
            original_path = os.path.normpath(original_path)

            # Add long path prefix if necessary
            original_path = add_long_path_prefix(original_path)

            # Check if the file still exists before proceeding
            if not os.path.exists(original_path):
                print(f"Skipping: File not found: {original_path}")
                continue

            # Create a cleaned filename
            cleaned_name = re.sub(r'[^a-zA-Z0-9\s._-]', '', file)  # Allow alphanumeric, spaces, periods, hyphens, and underscores
            cleaned_name = cleaned_name.replace(';', '')  # Remove semicolons

            # Ensure filename doesn't exceed max length
            base, ext = os.path.splitext(cleaned_name)
            max_base_length = max_length - len(ext)

            if len(base) > max_base_length:
                base = base[:max_base_length]  # Truncate the base name

            cleaned_name = f"{base}{ext}"
            cleaned_path = os.path.join(root, cleaned_name)

            # Add long path prefix for the cleaned path
            cleaned_path = add_long_path_prefix(cleaned_path)

            # Ensure the cleaned name is unique
            if os.path.exists(cleaned_path):
                counter = 1
                while os.path.exists(cleaned_path):
                    truncated_base = base[:max_base_length - len(f"_{counter}")]
                    cleaned_name = f"{truncated_base}_{counter}{ext}"
                    cleaned_path = os.path.join(root, cleaned_name)
                    cleaned_path = add_long_path_prefix(cleaned_path)
                    counter += 1

            # Rename file if the name changes
            if original_path != cleaned_path:
                print(f"Renaming: {original_path} -> {cleaned_path}")
                try:
                    os.rename(original_path, cleaned_path)
                except FileNotFoundError:
                    print(f"Error: File not found during renaming: {original_path}")
                except OSError as e:
                    print(f"Error: OS error during renaming: {e}")
                except Exception as e:
                    print(f"Error: {e}")

if __name__ == "__main__":
    # Enable long path support for Windows if applicable
    if sys.platform == 'win32':
        import ctypes
        try:
            ctypes.windll.kernel32.SetDllDirectoryW(None)
        except Exception:
            pass

    # Prompt user for the directory path
    directory_path = input("Enter the path of the directory to clean file names: ").strip()

    # Check if the path exists
    if not os.path.isdir(directory_path):
        print("Error: The specified path does not exist or is not a directory.")
    else:
        # Prompt user for maximum length
        max_length = input("Enter the maximum filename length (default is 255): ").strip()
        max_length = int(max_length) if max_length.isdigit() else 255
        clean_file_names(directory_path, max_length=max_length)

