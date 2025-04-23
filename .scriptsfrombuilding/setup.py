import os
import subprocess
import sys
import shutil

# --- Configuration ---
VENV_DIR = ".venv"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(PROJECT_DIR, "database")
IMAGES_DIR = os.path.join(DATABASE_DIR, "images")
BEFORE_DIR = os.path.join(IMAGES_DIR, "before")
AFTER_DIR = os.path.join(IMAGES_DIR, "after")
RUN_APP_SCRIPT = os.path.join(PROJECT_DIR, "app.py")

# List of required packages
# Note: 'tensorflow' can be large. If you don't have a compatible GPU or want a smaller install,
# you might replace 'tensorflow' with 'tensorflow-cpu'.
REQUIRED_PACKAGES = ['Flask', 'deepface', 'numpy', 'scipy', 'tensorflow-macos']

# --- Helper Functions ---

def run_command(command, check=True, cwd=None, capture_output=False, text=False):
    """Runs a command using subprocess and handles errors."""
    print(f"\n>>> Running command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=check,
            cwd=cwd,
            capture_output=capture_output,
            text=text,
            # Recommended for better handling across platforms, especially with PATH
            shell=False
        )
        if capture_output:
            print(result.stdout)
            if result.stderr:
                print("--- STDERR ---", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                print("--------------", file=sys.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"!!! Command failed with error code {e.returncode}: {' '.join(command)}", file=sys.stderr)
        if e.stderr:
             print(f"!!! Error Output:\n{e.stderr}", file=sys.stderr)
        if e.stdout:
             print(f"!!! Standard Output:\n{e.stdout}", file=sys.stderr)
        sys.exit(1) # Exit script if a critical command fails
    except FileNotFoundError:
        print(f"!!! Error: Command not found. Make sure '{command[0]}' is installed and in your PATH.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"!!! An unexpected error occurred running command: {' '.join(command)}", file=sys.stderr)
        print(f"!!! Error details: {e}", file=sys.stderr)
        sys.exit(1)


def get_venv_executable(venv_dir, executable_name):
    """Gets the platform-specific path to an executable in the venv."""
    if sys.platform == "win32":
        path = os.path.join(venv_dir, "Scripts", f"{executable_name}.exe")
    else: # Linux, macOS
        path = os.path.join(venv_dir, "bin", executable_name)

    if not os.path.exists(path):
         # Fallback check if Scripts/bin structure isn't exactly as expected
         alt_path_win = os.path.join(venv_dir, "Scripts", executable_name) # Maybe .exe missing?
         alt_path_nix = os.path.join(venv_dir, "bin", f"{executable_name}.exe") # Unlikely but check

         if sys.platform == "win32" and os.path.exists(alt_path_win):
             return alt_path_win
         if sys.platform != "win32" and os.path.exists(alt_path_nix):
              return alt_path_nix

         print(f"!!! Error: Could not find '{executable_name}' executable at expected path: {path}", file=sys.stderr)
         # Try to find python in PATH as last resort, though might not be venv one
         found_in_path = shutil.which(executable_name)
         if found_in_path:
             print(f"--- Found '{executable_name}' in PATH: {found_in_path}. Attempting to use it, but it might not be from the correct venv.", file=sys.stderr)
             return found_in_path
         else:
             print(f"!!! Critical: Cannot proceed without finding '{executable_name}'.", file=sys.stderr)
             sys.exit(1)
    return path

# --- Main Setup Logic ---

print("--- Starting Setup and Run Script ---")

# 1. Check for app.py early
if not os.path.exists(RUN_APP_SCRIPT):
    print(f"!!! Error: The Flask application script '{os.path.basename(RUN_APP_SCRIPT)}' was not found in this directory.", file=sys.stderr)
    print(f"!!! Please make sure it exists at: {RUN_APP_SCRIPT}", file=sys.stderr)
    sys.exit(1)
else:
    print(f"--- Found Flask app script: {os.path.basename(RUN_APP_SCRIPT)}")


# 2. Handle Virtual Environment
venv_path = os.path.join(PROJECT_DIR, VENV_DIR)
if not os.path.exists(venv_path):
    print(f"--- Virtual environment '{VENV_DIR}' not found. Creating using python3.11...")
    # Use python3.11 explicitly to create the venv
    run_command(['python3.11', "-m", "venv", venv_path])
    print(f"--- Virtual environment created at: {venv_path}")
else:
    print(f"--- Virtual environment '{VENV_DIR}' already exists.")

# Get paths to python and pip within the venv
python_exe = get_venv_executable(venv_path, "python")
pip_exe = get_venv_executable(venv_path, "pip")
print(f"--- Using Python executable: {python_exe}")
print(f"--- Using Pip executable: {pip_exe}")

# 3. Install Libraries into venv
print(f"\n--- Installing required packages into '{VENV_DIR}'...")
install_command = [pip_exe, "install"] + REQUIRED_PACKAGES
# Add --upgrade for potentially getting newer compatible versions if needed
# install_command.insert(2, "--upgrade")
# Add --no-cache-dir to potentially save space/avoid cache issues
# install_command.insert(2, "--no-cache-dir")
run_command(install_command)
print("--- Package installation complete.")

# 4. Create Directory Structure
print("\n--- Ensuring directory structure exists...")
try:
    os.makedirs(BEFORE_DIR, exist_ok=True)
    print(f"--- Ensured directory exists: {BEFORE_DIR}")
    os.makedirs(AFTER_DIR, exist_ok=True)
    print(f"--- Ensured directory exists: {AFTER_DIR}")
    # Note: Uploads folder is created by app.py, no need to create it here
except OSError as e:
    print(f"!!! Error creating directories: {e}", file=sys.stderr)
    sys.exit(1)
print("--- Directory structure checked/created.")


# 5. Start the Flask Application using the venv's Python
print(f"\n--- === Starting the Flask Application ({os.path.basename(RUN_APP_SCRIPT)}) === ---")
print(f"--- Using Python from: {python_exe}")
print("--- Press CTRL+C to stop the server.")

# Run the app script using the Python interpreter from the virtual environment
# Use check=False because the app runs indefinitely and we want to see its output directly.
# Its exit code upon CTRL+C might not be 0.
run_command([python_exe, RUN_APP_SCRIPT], check=False)

print("\n--- Flask application stopped. ---")
print("--- Script finished. ---")