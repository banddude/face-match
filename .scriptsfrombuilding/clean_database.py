import os
import json
import logging
from deepface import DeepFace

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, 'database')
IMAGES_SUBDIR = os.path.join(DATABASE_PATH, 'images')
BEFORE_FOLDER = os.path.join(IMAGES_SUBDIR, 'before')
AFTER_FOLDER = os.path.join(IMAGES_SUBDIR, 'after')
METADATA_FILE = os.path.join(DATABASE_PATH, 'scraped_data.json')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MODEL_NAME = 'Facenet512' # Match the model used in run_app.py
DETECTOR_BACKEND = 'retinaface' # Match the detector used in run_app.py

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_database():
    logger.info("--- Starting Database Cleaning Process ---")
    logger.warning("!!! This script will permanently delete images and modify JSON data. !!!")

    if not os.path.exists(BEFORE_FOLDER):
        logger.error(f"'Before' directory not found: {BEFORE_FOLDER}")
        return

    if not os.path.exists(METADATA_FILE):
        logger.error(f"Metadata JSON file not found: {METADATA_FILE}")
        return

    ids_to_remove = []
    files_processed = 0
    files_failed_detection = 0

    logger.info(f"Scanning files in: {BEFORE_FOLDER}")
    for filename in os.listdir(BEFORE_FOLDER):
        if allowed_file(filename):
            files_processed += 1
            before_path_full = os.path.join(BEFORE_FOLDER, filename)
            logger.info(f"Checking {filename}...")

            try:
                # Attempt face detection - enforce detection is True here!
                _ = DeepFace.represent(img_path=before_path_full,
                                       model_name=MODEL_NAME,
                                       enforce_detection=True, # Force detection check
                                       detector_backend=DETECTOR_BACKEND)
                # logger.info(f"Face detected in {filename}.") # Optional: uncomment for verbose success logs

            except ValueError as e:
                # Check if the error message specifically indicates no face was detected
                if "Face could not be detected" in str(e):
                    logger.warning(f"No face detected in {filename}. Flagging for removal.")
                    files_failed_detection += 1
                    case_id = os.path.splitext(filename)[0]
                    ids_to_remove.append(case_id)

                    # Delete corresponding images
                    after_path_full = os.path.join(AFTER_FOLDER, filename)

                    try:
                        logger.info(f"Deleting 'before' image: {before_path_full}")
                        os.remove(before_path_full)
                    except OSError as delete_error:
                        logger.error(f"Error deleting {before_path_full}: {delete_error}")

                    if os.path.exists(after_path_full):
                        try:
                            logger.info(f"Deleting 'after' image: {after_path_full}")
                            os.remove(after_path_full)
                        except OSError as delete_error:
                            logger.error(f"Error deleting {after_path_full}: {delete_error}")
                    else:
                         logger.info(f"No corresponding 'after' image found for {filename} (already deleted or never existed)." )

                else:
                    # Different ValueError occurred
                    logger.error(f"Unexpected ValueError processing {filename}: {e}")
            except Exception as e:
                # Catch any other exceptions during DeepFace processing
                logger.error(f"Error processing {filename}: {e}", exc_info=True)

        # else: # Optional: uncomment for verbose skip logs
        #     logger.debug(f"Skipping non-allowed file: {filename}")

    logger.info(f"Finished scanning {files_processed} allowed files. Found {files_failed_detection} with no detectable face.")

    # Clean the JSON file
    if ids_to_remove:
        logger.info(f"Removing entries for {len(ids_to_remove)} case IDs from {os.path.basename(METADATA_FILE)}: {ids_to_remove}")
        try:
            with open(METADATA_FILE, 'r') as f:
                metadata_list = json.load(f)

            # Filter out the entries to remove
            cleaned_metadata = [item for item in metadata_list if item.get('case_id') not in ids_to_remove]

            # Write the cleaned data back
            with open(METADATA_FILE, 'w') as f:
                json.dump(cleaned_metadata, f, indent=4) # Use indent for readability

            logger.info(f"Successfully updated {os.path.basename(METADATA_FILE)}.")

        except Exception as e:
            logger.error(f"Failed to update metadata file {METADATA_FILE}: {e}", exc_info=True)
    else:
        logger.info("No entries needed removal from JSON file.")

    logger.info("--- Database Cleaning Process Finished ---")

if __name__ == '__main__':
    # Add a small delay or confirmation perhaps?
    # input("Press Enter to start the cleaning process (WARNING: DELETES FILES)...")
    clean_database() 