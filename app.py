import os
import time
from flask import Flask, request, jsonify, render_template_string, send_from_directory, url_for
from werkzeug.utils import secure_filename
import numpy as np
from scipy.spatial.distance import cosine
from deepface import DeepFace
import logging
import threading
import json
import pickle  # For saving/loading the database cache

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
DATABASE_PATH = 'database'
IMAGES_SUBDIR = 'images'
BEFORE_FOLDER_NAME = 'before'
AFTER_FOLDER_NAME = 'after'
DB_PICKLE_FILE = os.path.join(DATABASE_PATH, 'face_db.pkl') # Cache file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MODEL_NAME = 'Facenet512' # Or 'ArcFace', 'VGG-Face' etc. - Ensure consistency!
SIMILARITY_THRESHOLD = 0.50 # Adjust as needed (Cosine Similarity: 1 is identical, 0 is unrelated)

# Construct full paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_UPLOAD_FOLDER = os.path.join(BASE_DIR, UPLOAD_FOLDER)
FULL_DATABASE_IMAGES_PATH = os.path.join(BASE_DIR, DATABASE_PATH, IMAGES_SUBDIR)
FULL_BEFORE_FOLDER = os.path.join(FULL_DATABASE_IMAGES_PATH, BEFORE_FOLDER_NAME)
FULL_AFTER_FOLDER = os.path.join(FULL_DATABASE_IMAGES_PATH, AFTER_FOLDER_NAME)

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = FULL_UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB upload limit

# --- In-Memory Database ---
# Stores {'before_path': path, 'after_path': path_or_none, 'embedding': np.array, 'metadata': dict}
face_database = []
metadata_mapping = {}  # mapping case_id to metadata dict
database_loaded = False
db_load_lock = threading.Lock()  # To prevent concurrent modification during loading

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_corresponding_after_image(before_filename_full_path):
    """Finds the corresponding 'after' image path based on the 'before' filename (same name)."""
    # For current naming, before and after share the same filename
    base_filename = os.path.basename(before_filename_full_path)
    after_path = os.path.join(FULL_AFTER_FOLDER, base_filename)
    if os.path.exists(after_path):
        return after_path
    else:
        logger.warning(f"No matching 'after' image found for '{base_filename}'")
        return None

def load_database():
    """Loads images from the database folder, computes embeddings, and stores them."""
    global face_database, database_loaded, metadata_mapping
    with db_load_lock:
        if database_loaded: # Avoid reloading if already done
             logger.info("Database already loaded.")
             return

        # Load metadata JSON
        metadata_file = os.path.join(BASE_DIR, DATABASE_PATH, 'scraped_data.json')
        try:
            with open(metadata_file, 'r') as f:
                metadata_list = json.load(f)
            metadata_mapping = {item['case_id']: item for item in metadata_list}
            logger.info(f"Loaded metadata for {len(metadata_mapping)} cases")
        except Exception as e:
            logger.warning(f"Could not load metadata JSON: {e}")
            metadata_mapping = {}

        logger.info(f"Loading database from: {FULL_BEFORE_FOLDER}")
        start_time = time.time()
        temp_database = []
        processed_files = 0
        failed_files = 0

        if not os.path.exists(FULL_BEFORE_FOLDER):
            logger.error(f"Before images folder not found: {FULL_BEFORE_FOLDER}")
            database_loaded = True # Mark as 'loaded' to prevent retries, even though it's empty
            return

        for filename in os.listdir(FULL_BEFORE_FOLDER):
            if allowed_file(filename):
                before_path_full = os.path.join(FULL_BEFORE_FOLDER, filename)
                try:
                    # Try to generate embedding even if face detection is weak
                    embedding_objs = DeepFace.represent(img_path=before_path_full,
                                                        model_name=MODEL_NAME,
                                                        enforce_detection=False, # Changed to False
                                                        detector_backend='retinaface')

                    if embedding_objs and len(embedding_objs) > 0:
                        embedding = embedding_objs[0]['embedding']

                        # Find corresponding 'after' image
                        after_path_full = find_corresponding_after_image(before_path_full)

                        # Store relative paths for URL generation later
                        before_path_relative = os.path.join(BEFORE_FOLDER_NAME, filename)
                        after_path_relative = os.path.join(AFTER_FOLDER_NAME, os.path.basename(after_path_full)) if after_path_full else None

                        # Attach metadata based on case_id
                        case_id = os.path.splitext(filename)[0]
                        entry_metadata = metadata_mapping.get(case_id)
                        temp_database.append({
                            'before_path': before_path_relative,
                            'after_path': after_path_relative,
                            'embedding': np.array(embedding), # Ensure it's a NumPy array
                            'metadata': entry_metadata
                        })
                        processed_files += 1
                        if processed_files % 10 == 0:
                             logger.info(f"Processed {processed_files} images...")
                    else:
                         logger.warning(f"No face detected or embedding failed for: {filename}")
                         failed_files += 1

                except ValueError as e:
                    logger.warning(f"ValueError (likely no face detected) for {filename}: {e}")
                    failed_files += 1
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}", exc_info=True)
                    failed_files += 1
            else:
                 logger.debug(f"Skipping non-allowed file: {filename}")

        face_database = temp_database
        database_loaded = True
        end_time = time.time()
        logger.info(f"Database loading complete. Processed: {processed_files}, Failed: {failed_files}. Time: {end_time - start_time:.2f} seconds.")
        if not face_database:
            logger.warning("Database is empty after loading process.")

        # Save the loaded database to a pickle file for caching
        cache_path = os.path.join(BASE_DIR, DB_PICKLE_FILE)
        try:
            logger.info(f"Saving database cache to {cache_path}...")
            with open(cache_path, 'wb') as f:
                pickle.dump(face_database, f)
            logger.info("Database cache saved successfully.")
        except Exception as e:
            logger.error(f"Error saving database cache: {e}", exc_info=True)


# --- HTML Template ---
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Facial Similarity Matcher</title>
    <style>
        /* Base Styles & Fonts (Approximation) */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

        :root {
            --color-bg-start: #1f2937; /* gray-900 */
            --color-bg-via: #581c87; /* purple-900 */
            --color-bg-end: #000000; /* black */
            --color-text-light: #d1d5db; /* gray-300 */
            --color-text-dim: #9ca3af; /* gray-400 */
            --color-text-very-dim: #6b7280; /* gray-500 */
            --color-accent-start: #60a5fa; /* blue-400 */
            --color-accent-end: #a78bfa; /* purple-400 */
            --color-error: #f87171; /* red-400 */
            --color-error-border: rgba(248, 113, 113, 0.5); /* red-500/50 */
            --color-card-bg: rgba(255, 255, 255, 0.05);
            --color-card-border: rgba(255, 255, 255, 0.1);
            --color-upload-border: #4b5563; /* gray-700 */
            --color-upload-border-hover: rgba(167, 139, 250, 0.5); /* purple-500/50 */
            --color-upload-bg-hover: rgba(0, 0, 0, 0.2);
            --color-button-bg-start: #a855f7; /* purple-500 */
            --color-button-bg-end: #3b82f6; /* blue-500 */
            --color-button-bg-start-hover: #9333ea; /* purple-600 */
            --color-button-bg-end-hover: #2563eb; /* blue-600 */
            --color-retry-bg: rgba(55, 65, 81, 0.5); /* gray-700/50 */
            --color-retry-border: #4b5563; /* gray-700 */
            --color-retry-hover-bg: #4b5563; /* gray-700 */
            --color-retry-hover-text: #ffffff;
            --suggestion-bg: rgba(0, 0, 0, 0.2);
        }

        body {
            font-family: 'Montserrat', sans-serif;
            margin: 0;
            min-height: 100vh;
            background-image: linear-gradient(to bottom right, var(--color-bg-start), var(--color-bg-via), var(--color-bg-end));
            color: var(--color-text-light);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem; /* p-4 */
        }

        /* Container & Card */
        .main-container {
            max-width: 56rem; /* max-w-4xl */
            width: 100%;
            margin-left: auto;
            margin-right: auto;
        }

        .card {
            background-color: var(--color-card-bg);
            backdrop-filter: blur(10px);
            border-radius: 0.75rem; /* rounded-xl */
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); /* shadow-lg */
            border: 1px solid var(--color-card-border);
            padding: 1.5rem; /* p-6 */
            margin-top: 2rem; /* space-y-8 between title and card */
        }
        @media (min-width: 640px) {
            .card { padding: 2rem; } /* md:p-8 */
        }

        /* Typography & Header */
        .header-section { text-align: center; margin-bottom: 2rem; /* space-y-4 approximated */ }
        .main-title {
            font-size: 1.875rem; /* text-3xl */
            font-weight: 700; /* font-bold */
            background-image: linear-gradient(to right, var(--color-accent-start), var(--color-accent-end));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 1rem;
        }
        .description { color: var(--color-text-dim); font-size: 0.875rem; /* text-sm */ margin-bottom: 0.5rem; }
        .info-note { display: flex; align-items: center; justify-content: center; gap: 0.5rem; color: var(--color-accent-start); font-size: 0.75rem; /* text-xs */}
        .info-note svg { width: 1rem; height: 1rem; }

        @media (min-width: 640px) { /* sm */
             .main-title { font-size: 2.25rem; } /* sm:text-4xl */
             .description { font-size: 1rem; } /* sm:text-base */
        }
         @media (min-width: 768px) { /* md */
             .main-title { font-size: 3rem; } /* md:text-5xl */
        }

        /* Upload Area */
        .upload-area {
            position: relative;
            border-radius: 0.5rem; /* rounded-lg */
            border: 2px dashed var(--color-upload-border);
            cursor: pointer;
            transition: border-color 0.2s ease-in-out, background-color 0.2s ease-in-out;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 1.5rem 1rem; /* py-6 px-4 */
            text-align: center;
            margin-bottom: 1.5rem; /* space-y-6 between sections approx */
        }
        .upload-area:hover {
            border-color: var(--color-upload-border-hover);
            background-color: var(--color-upload-bg-hover);
        }
        .upload-area.error-state {
            border-color: var(--color-error-border);
        }
        .upload-area svg {
            width: 2.5rem; /* w-10 */
            height: 2.5rem; /* h-10 */
            color: var(--color-text-dim);
            margin-bottom: 0.5rem; /* mb-2 */
        }
         .upload-area.error-state svg { color: var(--color-error); }
        .upload-area .upload-text { font-size: 0.875rem; color: var(--color-text-dim); }
        .upload-area.error-state .upload-text { color: var(--color-error); }
        .upload-area .upload-hint { font-size: 0.75rem; color: var(--color-text-very-dim); }
        .upload-area.error-state .upload-hint { color: var(--color-error); opacity: 0.8; }
        .upload-area .selected-file-name {
            margin-top: 1rem; /* mt-4 */
            font-size: 0.75rem; /* text-xs */
            color: var(--color-text-dim);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 200px;
        }
         @media (min-width: 640px) { /* sm */
            .upload-area .selected-file-name { max-width: 24rem; } /* sm:max-w-xs approx */
         }

        /* Hidden File Input */
        .hidden-input { display: none; }

        /* Error Message */
        .error-message {
            display: flex;
            align-items: center;
            gap: 0.5rem; /* gap-2 */
            color: var(--color-error);
            font-size: 0.875rem; /* text-sm */
            margin-top: 1rem; /* Added margin if needed */
            margin-bottom: 1rem; /* space-y-6 between sections approx */
        }
        .error-message svg { width: 1.25rem; height: 1.25rem; } /* w-5 h-5 */

        /* Process Button */
        .process-button {
            width: 100%;
            background-image: linear-gradient(to right, var(--color-button-bg-start), var(--color-button-bg-end));
            color: white;
            padding: 0.75rem 0; /* py-3 */
            font-weight: 600; /* font-semibold */
            font-size: 1.125rem; /* text-lg */
            border: none;
            border-radius: 0.375rem; /* rounded-md implicitly by Button component */
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* shadow-lg */
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem; /* mr-2 equivalent */
        }
        .process-button:hover {
            background-image: linear-gradient(to right, var(--color-button-bg-start-hover), var(--color-button-bg-end-hover));
             box-shadow: 0 0 15px 0 var(--color-button-bg-start); /* hover:shadow-purple-500/50 approx */
        }
        .process-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            background-image: linear-gradient(to right, var(--color-button-bg-start), var(--color-button-bg-end)); /* Keep gradient when disabled */
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .loader {
            width: 1.25rem; /* w-5 */
            height: 1.25rem; /* h-5 */
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-left-color: white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Results Section */
        .results-section {
            margin-top: 1.5rem; /* space-y-6 approx */
        }
        .results-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem; /* gap-6 */
            margin-bottom: 1.5rem; /* space-y-6 approx */
        }
        @media (min-width: 768px) { /* md */
            .results-grid { grid-template-columns: 1fr 1fr 1fr; } /* Adjust to 3 columns: Uploaded, Before, After */
        }
        .image-box { text-align: center; }
        .image-box h3 { /* was h2 before */
            font-size: 1.125rem; /* text-lg */
            font-weight: 600; /* font-semibold */
            color: var(--color-text-light);
            margin-bottom: 0.5rem; /* space-y-2 approx */
        }
        .image-box img {
            max-width: 100%;
            height: auto;
            object-fit: cover;
            border-radius: 0.5rem; /* rounded-lg */
            border: 1px solid var(--color-upload-border); /* border-gray-700 */
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* shadow-md */
        }

        /* Metadata & Score */
         .metadata-section {
            margin-bottom: 1.5rem; /* space-y-6 approx */
            margin-bottom: 3rem; /* Increased bottom margin */
         }
        .metadata-section h3 {
            font-size: 1.25rem; /* text-xl */
            font-weight: 600; /* font-semibold */
            color: var(--color-accent-start); /* text-blue-300 approx */
            margin-bottom: 1rem; /* space-y-4 approx */
        }
        .metadata-item {
            background-color: var(--suggestion-bg);
            padding: 1rem; /* p-4 */
            border-radius: 0.5rem; /* rounded-lg */
            border: 1px solid var(--color-upload-border); /* border-gray-700 */
            color: var(--color-text-light);
            margin-bottom: 0.75rem; /* Space between items */
            font-size: 0.875rem;
        }
        .metadata-item strong { color: #cbd5e1; } /* gray-300 slight emphasis */
        .metadata-item a { color: var(--color-accent-start); text-decoration: none; }
        .metadata-item a:hover { text-decoration: underline; }
        .similarity-score {
             font-weight: 600;
             font-size: 1.1em;
             text-align: center;
             margin-top: 1rem;
             color: var(--color-text-light);
             /* Ensure visibility and avoid overlap */
             position: relative; /* Ensure positioning context */
             z-index: 1;         /* Try to bring it forward */
             height: auto;       /* Ensure it takes vertical space */
             width: 100%;        /* Take available width */
             padding: 0.5rem 0;  /* Add some padding */
        }

        /* Retry Button */
        .retry-section { text-align: center; margin-top: 1.5rem; }
        .retry-button {
            background-color: var(--color-retry-bg);
            color: var(--color-text-light);
            border: 1px solid var(--color-retry-border);
            padding: 0.5rem 1.5rem; /* px-6 py-2 */
            border-radius: 9999px; /* rounded-full */
            cursor: pointer;
            transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
            font-size: 0.875rem; /* Match button text size */
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06); /* shadow-md */
        }
        .retry-button:hover {
            background-color: var(--color-retry-hover-bg);
            color: var(--color-retry-hover-text);
             box-shadow: 0 0 10px 0 rgba(75, 85, 99, 0.5); /* hover:shadow-gray-700/50 approx */
        }

        /* Status Messages (like no match) */
        .status-message {
            margin-top: 1rem;
            padding: 0.75rem 1rem;
            border-radius: 0.375rem;
            text-align: center;
            font-size: 0.875rem;
            border: 1px solid transparent;
        }
        .status-message.error {
             background-color: rgba(248, 113, 113, 0.1); /* Lighter red bg */
             color: var(--color-error);
             border-color: var(--color-error-border);
             display: flex;
             align-items: center;
             justify-content: center;
             gap: 0.5rem;
        }
        .status-message.error svg { width: 1.25rem; height: 1.25rem; }

        /* Utility */
        .hidden { display: none !important; }

        /* --- ADDED RULE --- */
        #uploadedImage {
            min-height: 50px; /* Ensure it takes up some space */
            display: block;   /* Reinforce block display */
            width: 100%;      /* Try forcing width too */
            object-fit: contain; /* Adjust object fit */
        }
        /* --- ADDED RULE for container --- */
        #uploadedImageBox {
             min-height: 60px; /* Ensure container has space */
             display: block !important; /* Force display override */
             width: 100%; /* Ensure container takes width */
        }
        /* --- REMOVED RULE for preview container --- */
        /* #imagePreviewArea { */
        /*     padding: 1rem; */
        /*     border: 1px dashed var(--color-text-very-dim); */
        /*     border-radius: 0.5rem; */
        /* } */

        /* --- NEW Styles for Top 3 Matches --- */
        .match-result-item {
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--color-card-border);
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }
        .match-result-item h4 {
             margin-top: 0;
             margin-bottom: 1rem;
             font-weight: 600;
             color: var(--color-accent-start);
        }
        .match-images {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap; /* Allow wrapping on smaller screens */
        }
        .match-images .image-box {
            flex: 1; /* Allow boxes to grow */
            min-width: 150px; /* Minimum width before wrapping */
        }
         .match-images .image-box img {
             max-height: 250px; /* Limit height of match images */
             width: 100%;
             object-fit: cover;
         }
        .match-metadata {
            margin-top: 1rem;
            font-size: 0.8rem;
            border-top: 1px solid var(--color-upload-border);
            padding-top: 1rem;
        }
        .match-metadata h5 {
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: var(--color-text-dim);
            font-size: 0.9em;
        }
        .match-metadata p {
            margin: 0.3rem 0;
            color: var(--color-text-light);
        }
         .match-metadata a {
            color: var(--color-accent-start);
            text-decoration: none;
        }
        .match-metadata a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="main-container">

        <div class="header-section">
            <h1 class="main-title">Face Comparison & Analysis</h1>
            <p class="description">
                Upload a photo to see similar before & after transformations and potential treatment suggestions based on matched cases.
            </p>
            <div class="info-note">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                <span>Please upload a clear, front-facing photo.</span>
            </div>
        </div>

        <div class="card">
            <div class="upload-section">
                 <input type="file" id="fileInput" class="hidden-input" accept="image/png, image/jpeg, image/jpg, image/webp">
                 <div id="uploadArea" class="upload-area" role="button" tabindex="0" aria-label="Upload image area">
                     <svg id="uploadIcon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7"></path><line x1="16" y1="5" x2="22" y2="5"></line><line x1="19" y1="2" x2="19" y2="8"></line><circle cx="9" cy="9" r="2"></circle><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"></path></svg>
                     <p class="upload-text" id="uploadText">Click to upload or drag and drop</p>
                     <p class="upload-hint" id="uploadHint">PNG, JPG, WEBP up to 5MB</p>
                     <div id="fileName" class="selected-file-name hidden"></div>

                     <div class="image-box" id="uploadedImageBox" style="display: none; width: 100%; margin-top: 1rem;">
                        <img id="uploadedImage" src="#" alt="Uploaded Image Preview" style="max-height: 200px;">
                     </div>

                 </div>
                 <div id="errorMessage" class="error-message hidden">
                     <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                     <span id="errorText"></span>
                 </div>
                 <button id="processButton" class="process-button">
                     <span id="buttonText">Process Image</span>
                     <div id="loader" class="loader hidden"></div>
                 </button>
            </div>

            <div id="resultsSection" class="results-section" style="display: none;"> 
                <div id="resultsGrid" class="results-grid">
                    <!-- Content will be added dynamically by JS -->
                </div>

                <!-- REMOVED static beforeImageBox -->
                <!-- REMOVED static afterImageBox -->
                <!-- REMOVED static metadataSection -->
                <!-- REMOVED static similarityScore -->
                <!-- REMOVED static noAfterMessage -->
                 <div id="noAfterMessage" class="status-message error" style="display: none;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                    <span>Match found, but the corresponding 'After' image is missing or not applicable for this case.</span>
                 </div>

                <div class="retry-section">
                    <button id="retryButton" class="retry-button">Try Again with a New Photo</button>
                </div>
            </div>

            <!-- No Match message moved outside results section if needed, or handled by JS -->
            <div id="noMatchMessage" class="status-message error" style="display: none;">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18.36 6.64a9 9 0 1 1-12.73 0"></path><line x1="12" y1="2" x2="12" y2="12"></line></svg> <span>No suitable match found in our database. Please try a different image.</span>
            </div>

        </div> </div> <script>
        // --- DOM Elements ---
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const uploadIcon = document.getElementById('uploadIcon');
        const uploadText = document.getElementById('uploadText');
        const uploadHint = document.getElementById('uploadHint');
        const fileNameDisplay = document.getElementById('fileName');
        const errorMessageDiv = document.getElementById('errorMessage');
        const errorTextSpan = document.getElementById('errorText');
        const processButton = document.getElementById('processButton');
        const buttonText = document.getElementById('buttonText');
        const loader = document.getElementById('loader');
        const resultsSection = document.getElementById('resultsSection');
        const resultsGrid = document.getElementById('resultsGrid'); // Get the grid itself
        const uploadedImageBox = document.getElementById('uploadedImageBox');
        const uploadedImage = document.getElementById('uploadedImage');
        const afterImage = document.getElementById('afterImage'); // Keep ref? Might be removed
        // const metadataSection = document.getElementById('metadataSection'); // Removed from HTML
        // const metadataList = document.getElementById('metadataList'); // Removed from HTML
        // const similarityScoreDiv = document.getElementById('similarityScore'); // Removed from HTML
        const noAfterMessage = document.getElementById('noAfterMessage'); // Kept for now, handle display in JS
        const noMatchMessage = document.getElementById('noMatchMessage');
        const retryButton = document.getElementById('retryButton');

        let currentFile = null;

        // --- Functions ---
        function showLoading(isLoading) {
            if (isLoading) {
                buttonText.textContent = 'Processing...';
                loader.classList.remove('hidden');
                processButton.disabled = true;
            } else {
                buttonText.textContent = 'Process Image';
                loader.classList.add('hidden');
                processButton.disabled = false;
            }
        }

        function showError(message) {
            if (message) {
                errorTextSpan.textContent = message;
                errorMessageDiv.classList.remove('hidden');
                uploadArea.classList.add('error-state');
                uploadIcon.style.color = 'var(--color-error)'; // Change icon color directly
            } else {
                errorMessageDiv.classList.add('hidden');
                uploadArea.classList.remove('error-state');
                 uploadIcon.style.color = 'var(--color-text-dim)'; // Revert icon color
            }
             // Hide specific messages when showing a general error
             noMatchMessage.style.display = 'none';
             noAfterMessage.style.display = 'none';
        }

        function resetUI() {
            currentFile = null;
            fileInput.value = ''; // Clear the actual input
            fileNameDisplay.textContent = '';
            fileNameDisplay.classList.add('hidden');
            showError(null); // Clear errors

            // Show upload area elements
            uploadIcon.style.display = 'block';
            uploadText.style.display = 'block';
            uploadHint.style.display = 'block';
            // Hide preview image box using alternative CSS
            uploadedImageBox.style.visibility = 'hidden';
            uploadedImageBox.style.height = '0px';
            uploadedImageBox.style.overflow = 'hidden';
            uploadedImageBox.style.marginTop = '0px'; // Remove margin when hidden
            console.log('[DEBUG] resetUI: Upload area elements shown, preview hidden (visibility method).');

            // Hide main results area
            resultsSection.style.display = 'none';
            // Explicitly hide children of results section just in case
            // beforeImageBox.style.display = 'none';
            // afterImageBox.style.display = 'none';
            // metadataList.innerHTML = ''; // Clear previous metadata
            // metadataSection.style.display = 'none';
            // similarityScoreDiv.style.display = 'none';
            // similarityScoreDiv.textContent = '';
            noMatchMessage.style.display = 'none';
            noAfterMessage.style.display = 'none';
            // Keep animation class reset
            resultsSection.classList.remove('visible');
        }

        function displayResults(result) {
            console.log('[DEBUG] Displaying results (Top 3):', result);
            resultsGrid.innerHTML = ''; // Clear previous results
            noMatchMessage.style.display = 'none'; // Hide no match message
            noAfterMessage.style.display = 'none'; // Hide no after message (handled per match)

            if (result.success && result.matches && result.matches.length > 0) {
                let resultsHTML = '';
                result.matches.forEach((match, index) => {
                    // --- Metadata generation --- 
                    let metadataHTML = '';
                    const meta = match.metadata;
                    if (meta && Object.keys(meta).length > 0) {
                         if (meta.case_id) {
                            metadataHTML += `<p><strong>Case ID:</strong> ${meta.case_id}</p>`;
                         }
                         if (meta.procedure_name) {
                            metadataHTML += `<p><strong>Procedure:</strong> ${meta.procedure_name}</p>`;
                         }
                        // Add other potential metadata fields here following the pattern
                         if (meta.page_url) {
                            metadataHTML += `<p><a href="${meta.page_url}" target="_blank" rel="noopener noreferrer">View Procedure Details</a></p>`;
                         }
                         if (metadataHTML === '') { // If meta exists but has no known fields we check for
                             metadataHTML = `<p>Additional details available.</p>`;
                         }
                    } else {
                        metadataHTML = '<p>No additional details available.</p>';
                    }
                    // --- End Metadata generation ---

                    // --- HTML for one match item --- 
                    resultsHTML += `
                        <div class="match-result-item">
                            <h4>Match ${index + 1} (Similarity: ${(match.similarity * 100).toFixed(1)}%)</h4>
                            <div class="match-images">
                                <div class="image-box">
                                    <h3>Before</h3>
                                    <img src="${match.match_before_url}" alt="Before Match ${index + 1}" onerror="this.alt='Error loading image'; this.src='#';">
                                </div>
                                <div class="image-box">
                                    <h3>After</h3>
                                    ${match.match_after_url ? 
                                        `<img src="${match.match_after_url}" alt="After Match ${index + 1}" onerror="this.alt='Error loading image'; this.src='#';">` : 
                                        '<p style="color: var(--color-text-very-dim);">N/A</p>'}
                                </div>
                            </div>
                            <div class="match-metadata">
                                <h5>Details:</h5>
                                ${metadataHTML}
                            </div>
                        </div>
                    `;
                    // --- End HTML for one match item ---
                });

                resultsGrid.innerHTML = resultsHTML; // Add all generated HTML at once
                resultsSection.style.display = 'block'; // Show the main results section
                console.log(`[DEBUG] Displayed ${result.matches.length} matches.`);

            } else {
                // Handle case where success is true but matches is empty, or success is false
                console.log('[DEBUG] No matches found or error occurred.');
                resultsSection.style.display = 'none'; // Hide results section
                noMatchMessage.style.display = 'flex'; // Show the main no match message
            }
        }


        // --- Event Listeners ---
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                fileInput.click();
            }
        });

        fileInput.addEventListener('change', (e) => {
            console.log('[DEBUG] fileInput change event triggered.');
            const file = e.target.files ? e.target.files[0] : null;
            console.log('[DEBUG] Selected file object:', file);

            if (file) {
                console.log('[DEBUG] File selected. Starting validation.');
                // Validation
                if (!file.type.startsWith('image/')) {
                    console.error('[DEBUG] Validation failed: Not an image type.', file.type);
                    showError('Please select a valid image file (PNG, JPG, WEBP).');
                    fileInput.value = ''; // Clear the input
                    currentFile = null;
                    return;
                }
                if (file.size > 5 * 1024 * 1024) { // 5MB limit
                    console.error('[DEBUG] Validation failed: File size too large.', file.size);
                    showError('Image size exceeds 5MB limit.');
                    fileInput.value = ''; // Clear the input
                    currentFile = null;
                    return;
                }
                console.log('[DEBUG] Validation passed.');

                // Clear previous results & errors before setting new file
                console.log('[DEBUG] Clearing previous errors and results.');
                showError(null);
                // Use resetUI which now uses inline styles
                resetUI(); // Call reset which handles hiding with styles
                // Set currentFile *after* resetUI in case resetUI clears it
                currentFile = file;
                console.log('[DEBUG] currentFile set after resetUI:', currentFile.name);
                fileNameDisplay.classList.add('hidden'); // Keep hiding filename text
                console.log('[DEBUG] Filename display hidden.');


                // --- IMAGE PREVIEW ---
                console.log('[DEBUG] Setting up FileReader.');
                const reader = new FileReader();

                reader.onloadstart = function() {
                    console.log('[DEBUG] FileReader: onloadstart triggered.');
                };

                reader.onprogress = function(event) {
                     if (event.lengthComputable) {
                        const percentLoaded = Math.round((event.loaded / event.total) * 100);
                        console.log(`[DEBUG] FileReader: onprogress - ${percentLoaded}% loaded.`);
                     } else {
                        console.log('[DEBUG] FileReader: onprogress - loading (length not computable).');
                     }
                };

                reader.onload = function(event) {
                    console.log('[DEBUG] FileReader: onload triggered (In-Place Preview).');
                    try {
                        // 1. Set the image source
                        uploadedImage.src = event.target.result;
                        console.log('[DEBUG] In-Place: uploadedImage.src set.');

                        // 2. Hide upload area icon/text
                        uploadIcon.style.display = 'none';
                        uploadText.style.display = 'none';
                        uploadHint.style.display = 'none';
                        console.log('[DEBUG] In-Place: Upload icon/text hidden.');

                        // 3. Show the preview image box (inside upload area)
                        // uploadedImageBox.style.display = 'block';
                        uploadedImageBox.style.visibility = 'visible';
                        uploadedImageBox.style.height = 'auto';    // Restore auto height
                        uploadedImageBox.style.overflow = 'visible'; // Restore overflow
                        uploadedImageBox.style.marginTop = '1rem'; // Restore margin
                        console.log('[DEBUG] In-Place: uploadedImageBox shown (visibility method).');

                        // 4. Ensure main results section is hidden
                        resultsSection.style.display = 'none';
                        console.log('[DEBUG] In-Place: resultsSection hidden.');

                    } catch (innerError) {
                         console.error('[DEBUG] Error inside In-Place reader.onload:', innerError);
                         showError("Error displaying image preview.");
                    }
                }

                reader.onerror = function(event) {
                    console.error("[DEBUG] FileReader: onerror triggered.");
                    console.error("[DEBUG] FileReader error details:", event.target.error);
                    showError("Could not read the selected file.");
                };

                 reader.onloadend = function() {
                    console.log('[DEBUG] FileReader: onloadend triggered.');
                 };


                console.log('[DEBUG] Calling reader.readAsDataURL(file).');
                reader.readAsDataURL(file);
                console.log('[DEBUG] reader.readAsDataURL call finished (async operation started).');

            } else {
                console.log('[DEBUG] No file selected or file cleared.');
                resetUI();
            }
        });

        // Handle Drag and Drop (Optional Enhancement)
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--color-upload-border-hover)'; // Indicate droppable
        });
        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
             if (!uploadArea.classList.contains('error-state')) {
                uploadArea.style.borderColor = 'var(--color-upload-border)';
            } else {
                 uploadArea.style.borderColor = 'var(--color-error-border)';
            }
        });
         uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            if (!uploadArea.classList.contains('error-state')) {
                uploadArea.style.borderColor = 'var(--color-upload-border)';
            } else {
                 uploadArea.style.borderColor = 'var(--color-error-border)';
            }
            const file = e.dataTransfer.files ? e.dataTransfer.files[0] : null;
             if (file) {
                 // Manually trigger the change event logic after setting the file
                 fileInput.files = e.dataTransfer.files; // Important: Assign dropped files to input
                 const changeEvent = new Event('change');
                 fileInput.dispatchEvent(changeEvent);
            }
        });


        processButton.addEventListener('click', async () => {
            console.log('Process button clicked. currentFile:', currentFile); // DEBUG LINE
            if (!currentFile) {
                showError('Please select an image first.');
                return;
            }

            showError(null); // Clear any previous errors before processing
            // Hide specific messages/sections before processing starts
            noMatchMessage.style.display = 'none';
            noAfterMessage.style.display = 'none';
            // metadataSection.style.display = 'none';
            // similarityScoreDiv.style.display = 'none';
            // Keep the uploaded preview visible, but hide other results
            // beforeImageBox.style.display = 'none';
            // afterImageBox.style.display = 'none';

            const formData = new FormData();
            formData.append('file', currentFile); // Use 'file' as the key, matching the target template's expectation

            try {
                // **IMPORTANT:** Use the /match endpoint as required by the second template
                const response = await fetch('/match', {
                    method: 'POST',
                    body: formData,
                });

                const result = await response.json();

                if (result.success) {
                    displayResults(result);
                } else {
                    // Handle specific errors from backend based on target template's logic
                    if (result.error_type === 'NO_MATCH_FOUND') {
                        noMatchMessage.style.display = 'block';
                    } else {
                        // General error message from backend or default
                         showError(`Error: ${result.message || 'Processing failed.'}`);
                    }
                }

            } catch (error) {
                console.error('Fetch error:', error);
                showError('An unexpected network error occurred. Please try again.');
            } finally {
                showLoading(false);
            }
        });

        retryButton.addEventListener('click', resetUI);

        // --- Initial State --- 
        document.addEventListener('DOMContentLoaded', () => {
            console.log('[DEBUG] DOMContentLoaded fired. Setting timeout for initial resetUI.');
            setTimeout(() => {
                 console.log('[DEBUG] Timeout fired. Running initial resetUI.');
                 resetUI(); // Ensure clean state *after* DOM is loaded and a short delay
            }, 100); // 100ms delay
        });

    </script>
</body>
</html>
"""

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    if not database_loaded:
         # Could optionally trigger loading here if it failed on startup,
         # but typically it's better to ensure it loads on start.
         # Or show a 'loading database' message on the page.
         pass # Assuming load_database() was called at startup
    return render_template_string(HTML_TEMPLATE)

@app.route('/match', methods=['POST'])
def match_face():
    """Handles image upload, embedding generation, and matching."""
    if not database_loaded or not face_database:
         logger.warning("Match request received but database is not loaded or empty.")
         return jsonify({'success': False, 'message': 'Face database is not ready. Please wait or contact administrator.'})

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part in the request.'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file.'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        uploaded_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(uploaded_filepath)
            logger.info(f"File uploaded successfully: {uploaded_filepath}")

            # 1. Generate embedding for uploaded image
            try:
                # Use silent=True to reduce console spam
                embedding_objs = DeepFace.represent(img_path=uploaded_filepath,
                                                    model_name=MODEL_NAME,
                                                    enforce_detection=True,
                                                    detector_backend='retinaface') # Specify detector

                if not embedding_objs or len(embedding_objs) == 0:
                     raise ValueError("No face detected in uploaded image.")

                uploaded_embedding = np.array(embedding_objs[0]['embedding'])

            except ValueError as e:
                 logger.warning(f"Could not process uploaded file {filename}: {e}")
                 # Clean up uploaded file
                 if os.path.exists(uploaded_filepath):
                     os.remove(uploaded_filepath)
                 return jsonify({'success': False, 'message': f"Could not process image: {e}"})
            except Exception as e:
                 logger.error(f"Unexpected error processing uploaded file {filename}: {e}", exc_info=True)
                 if os.path.exists(uploaded_filepath):
                     os.remove(uploaded_filepath)
                 return jsonify({'success': False, 'message': 'An unexpected error occurred during image processing.'})


            # 2. Compare with database embeddings
            distances = []

            for i, db_entry in enumerate(face_database):
                try:
                    # Ensure db_entry['embedding'] is a valid numpy array
                    db_embedding = db_entry.get('embedding')
                    if db_embedding is None or not isinstance(db_embedding, np.ndarray):
                        logger.warning(f"Skipping entry {i} due to missing or invalid embedding.")
                        continue
                    if db_embedding.shape != uploaded_embedding.shape:
                         logger.warning(f"Skipping entry {i} due to incompatible embedding shape: {db_embedding.shape} vs {uploaded_embedding.shape}")
                         continue

                    distance = cosine(uploaded_embedding, db_embedding)
                    distances.append({'index': i, 'distance': distance})
                except Exception as dist_err:
                     logger.error(f"Error calculating distance for entry {i}: {dist_err}", exc_info=True)

            if not distances:
                 logger.warning(f"No valid distances calculated for {filename}. Database might have issues.")
                 if os.path.exists(uploaded_filepath):
                      os.remove(uploaded_filepath)
                 return jsonify({'success': False, 'message': 'Could not compare with database entries.', 'error_type': 'DB_COMPARE_ERROR'})

            # Sort by distance (ascending)
            distances.sort(key=lambda x: x['distance'])

            # 3. Prepare results for top N matches
            num_matches_to_show = min(3, len(distances))
            top_matches_data = []

            logger.info(f"Found {len(distances)} potential matches. Preparing top {num_matches_to_show}.")

            for i in range(num_matches_to_show):
                match_info = distances[i]
                db_index = match_info['index']
                distance = match_info['distance']
                similarity = 1 - distance

                db_entry = face_database[db_index]
                logger.info(f"  Match {i+1}: Index {db_index}, Distance: {distance:.4f}, Similarity: {similarity:.4f}, Before: {db_entry.get('before_path')}")

                # Generate URLs for the images
                before_filename = os.path.basename(db_entry['before_path']) if db_entry.get('before_path') else 'invalid'
                before_url = url_for('database_image', subdir=BEFORE_FOLDER_NAME, filename=before_filename)
                
                after_filename = os.path.basename(db_entry['after_path']) if db_entry.get('after_path') else None
                after_url = url_for('database_image', subdir=AFTER_FOLDER_NAME, filename=after_filename) if after_filename else None
                
                top_matches_data.append({
                    'similarity': similarity,
                    'match_before_url': before_url,
                    'match_after_url': after_url,
                    'metadata': db_entry.get('metadata', {})
                })

            # Return results
            uploaded_url = url_for('uploaded_file', filename=filename)
            if num_matches_to_show > 0:
                 logger.info(f"Returning {num_matches_to_show} matches for {filename}.")
                 return jsonify({
                    'success': True,
                    'message': f'{num_matches_to_show} matches found.',
                    'uploaded_image_url': uploaded_url,
                    'matches': top_matches_data # List of matches
                })
            else:
                 # This case should be rare if distances were calculated but empty database?
                 logger.info(f"No matches found in the database for {filename}.")
                 if os.path.exists(uploaded_filepath):
                     os.remove(uploaded_filepath)
                 return jsonify({
                    'success': False,
                    'message': 'No suitable matches found in the database.',
                    'error_type': 'NO_MATCH_FOUND'
                 })

        except Exception as e:
            logger.error(f"Error during matching process for {filename}: {e}", exc_info=True)
            # Clean up uploaded file in case of error
            if os.path.exists(uploaded_filepath):
                os.remove(uploaded_filepath)
            return jsonify({'success': False, 'message': 'An internal server error occurred.'})
    else:
        return jsonify({'success': False, 'message': 'File type not allowed.'})


# Routes to serve images securely
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves files from the upload folder."""
    # Security: Ensure filename is secure and path traversal is not possible
    return send_from_directory(app.config['UPLOAD_FOLDER'], secure_filename(filename))

@app.route('/database_images/<path:subdir>/<path:filename>')
def database_image(subdir, filename):
    """Serves images from the database/images/before or database/images/after folders."""
    # Security: Validate subdir and filename
    safe_subdir = secure_filename(subdir)
    safe_filename = secure_filename(filename)
    if safe_subdir not in [BEFORE_FOLDER_NAME, AFTER_FOLDER_NAME]:
        return "Invalid directory", 404

    directory = os.path.join(FULL_DATABASE_IMAGES_PATH, safe_subdir)
    logger.debug(f"Serving database image from: {directory}, file: {safe_filename}")
    # Use send_from_directory for security (handles path traversal)
    return send_from_directory(directory, safe_filename)


# --- Main Execution ---
if __name__ == '__main__':
    # Ensure necessary folders exist
    if not os.path.exists(FULL_UPLOAD_FOLDER):
        os.makedirs(FULL_UPLOAD_FOLDER)
        logger.info(f"Created upload folder: {FULL_UPLOAD_FOLDER}")
    if not os.path.exists(FULL_BEFORE_FOLDER):
        os.makedirs(FULL_BEFORE_FOLDER)
        logger.info(f"Created database 'before' folder: {FULL_BEFORE_FOLDER}")
    if not os.path.exists(FULL_AFTER_FOLDER):
        os.makedirs(FULL_AFTER_FOLDER)
        logger.info(f"Created database 'after' folder: {FULL_AFTER_FOLDER}")

    # Attempt to load database from cache first
    cache_file_path = os.path.join(BASE_DIR, DB_PICKLE_FILE)
    loaded_from_cache = False
    if os.path.exists(cache_file_path):
        try:
            logger.info(f"Attempting to load database from cache: {cache_file_path}")
            with open(cache_file_path, 'rb') as f:
                face_database = pickle.load(f)
            database_loaded = True
            loaded_from_cache = True
            logger.info(f"Successfully loaded database from cache ({len(face_database)} entries).")
        except Exception as e:
            logger.warning(f"Failed to load database from cache: {e}. Will reload from source.")
            database_loaded = False # Ensure it's false if cache load fails

    # If not loaded from cache, start background loading
    if not loaded_from_cache:
        logger.info("Starting database loading in background thread...")
        # Ensure metadata_mapping is reset if we are reloading from scratch
        metadata_mapping = {}
        db_thread = threading.Thread(target=load_database, daemon=True)
        db_thread.start()
    else:
        # If loaded from cache, we still need the metadata_mapping for lookups
        # Note: This assumes metadata doesn't change independently of images needing reprocessing
        # A more robust solution would store metadata_mapping in cache too or reload it here.
        metadata_file = os.path.join(BASE_DIR, DATABASE_PATH, 'scraped_data.json')
        try:
            with open(metadata_file, 'r') as f:
                metadata_list = json.load(f)
            metadata_mapping = {item['case_id']: item for item in metadata_list}
            logger.info(f"Loaded metadata mapping for cached database ({len(metadata_mapping)} cases).")
        except Exception as e:
            logger.warning(f"Could not load metadata JSON for cached DB: {e}")
            metadata_mapping = {}

    # Start the Flask server immediately
    logger.info("Starting Flask server...")
    # Set host='0.0.0.0' to make it accessible on your network
    # Use use_reloader=True to enable auto-reloading
    app.run(debug=True, host='0.0.0.0', port=3000, use_reloader=True)