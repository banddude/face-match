# Face Match Application

## Description

This Flask application allows users to upload a facial image and find the top 3 most similar faces from a pre-defined database of before/after image pairs. It displays the matches along with associated metadata.

## Setup Instructions

### Prerequisites

*   Python 3.11
*   Git (for cloning)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/banddude/face-match.git
    cd face-match
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    ```
    *(On Windows, use `.venv\Scripts\activate`)*

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    **Note:** The `requirements.txt` file reflects the exact state of the original working development environment (macOS ARM). Due to complex interactions between `tensorflow`, `tensorflow-macos`, and other libraries, installing these requirements into a fresh environment might fail with dependency conflicts. If you encounter issues, you may need to manually adjust versions or install `tensorflow` appropriate for your specific platform.

## Running the Application

1.  **Ensure Database:** Make sure the `database/` directory contains:
    *   `images/before/` sub-directory with before images.
    *   `images/after/` sub-directory with corresponding after images (named identically to their 'before' counterparts).
    *   `scraped_data.json` file containing metadata linked by case ID (filename without extension).
    *(The database cache `face_db.pkl` will be generated on first run if not present)*

2.  **Run the Flask app:**
    ```bash
    python app.py
    ```

3.  **Access the application:** Open your web browser and navigate to `http://localhost:3000` (or the address provided in the terminal). 