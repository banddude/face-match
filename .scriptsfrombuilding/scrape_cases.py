import requests
from bs4 import BeautifulSoup
import json
import time
import logging
import os # Added import
from urllib.parse import urljoin, urlparse # Added urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DB_DIR = os.path.join(BASE_DIR, 'database', 'images')
BEFORE_IMG_DIR = os.path.join(IMAGE_DB_DIR, 'before')
AFTER_IMG_DIR = os.path.join(IMAGE_DB_DIR, 'after')
OUTPUT_JSON_FILE = os.path.join(BASE_DIR, 'database', 'scraped_data.json')

# Ensure directories exist
os.makedirs(BEFORE_IMG_DIR, exist_ok=True)
os.makedirs(AFTER_IMG_DIR, exist_ok=True)

def get_image_url(img_container):
    """Extracts the primary image URL from an img tag, handling picture/source tags."""
    picture = img_container.find('picture')
    if picture:
        source_webp = picture.find('source', type='image/webp')
        if source_webp and source_webp.has_attr('srcset'):
            # Prefer webp if available
            return source_webp['srcset'].split(',')[0].split(' ')[0] # Get first URL if multiple densities
        img = picture.find('img')
        if img and img.has_attr('src'):
            return img['src']
    # Fallback for direct img tag
    img = img_container.find('img')
    if img and img.has_attr('src'):
        return img['src']
    return None

def scrape_patient_info(soup):
    """Extracts patient details from the info list."""
    info = {}
    info_div = soup.find('div', id='patient-info')
    if not info_div:
        logging.warning("Patient info div not found.")
        return info

    list_items = info_div.find_all('li')
    for item in list_items:
        text = item.get_text(strip=True)
        if ':' in text:
            key, value = text.split(':', 1)
            key_clean = key.strip().lower().replace(' ', '_').replace('#', 'id')
            if key_clean == 'procedure':
                 # Extract procedure name and link if available
                link_tag = item.find('a')
                if link_tag:
                    info['procedure_name'] = link_tag.get_text(strip=True)
                    info['procedure_link'] = link_tag.get('href') if link_tag.has_attr('href') else None
                else:
                     info['procedure_name'] = value.strip()
                     info['procedure_link'] = None
            else:
                info[key_clean] = value.strip()
    return info

def download_image(url, folder, filename_base, session):
    """Downloads an image from a URL into the specified folder."""
    if not url:
        logging.warning(f"No URL provided for {filename_base}, skipping download.")
        return
    try:
        response = session.get(url, stream=True, timeout=20)
        response.raise_for_status()

        # Get file extension
        parsed_url = urlparse(url)
        _, ext = os.path.splitext(parsed_url.path)
        if not ext: # Basic fallback if path has no extension
            content_type = response.headers.get('content-type')
            if content_type and '/' in content_type:
                ext = '.' + content_type.split('/')[1].split(';')[0] # e.g. .jpeg from image/jpeg; charset=UTF-8
            else:
                ext = '.jpg' # Default extension
        
        filename = f"{filename_base}{ext}"
        filepath = os.path.join(folder, filename)

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Successfully downloaded {url} to {filepath}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading image {url}: {e}")
    except IOError as e:
        logging.error(f"Error saving image {url} to {filepath}: {e}")

def scrape_case_page(url, session):
    """Scrapes data from a single case page and downloads images."""
    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None # Return None if page fetch fails

    soup = BeautifulSoup(response.text, 'html.parser')

    data = {}

    # Extract Case ID from heading or URL (more robust)
    case_id = None
    heading = soup.find('h1')
    if heading and 'Case #' in heading.text:
        try:
            case_id = heading.text.split('#')[-1].strip()
        except IndexError:
            pass # Fallback below

    if not case_id:
         # Try extracting from URL as fallback
         try:
            path_parts = url.strip('/').split('/')
            last_part = path_parts[-1].split('?')[0] # Remove query params if any
            if last_part.isdigit():
                case_id = last_part
         except Exception:
             pass # Logging handled later if still None

    if not case_id:
        logging.error(f"Could not determine Case ID for URL: {url}. Skipping image download and data saving.")
        return None # Cannot proceed without case ID

    data['case_id'] = case_id
    data['page_url'] = url

    # Extract image URLs
    before_image_url = None
    after_image_url = None
    image_pair_div = soup.find('div', class_='image-pair')
    if image_pair_div:
        img_boxes = image_pair_div.find_all('div', class_='img-box')
        if len(img_boxes) >= 2:
            before_image_url = get_image_url(img_boxes[0])
            after_image_url = get_image_url(img_boxes[1])
            data['before_image_url'] = before_image_url
            data['after_image_url'] = after_image_url
        else:
            logging.warning(f"Could not find two img-box divs for images on {url}")
            data['before_image_url'] = None
            data['after_image_url'] = None
    else:
        logging.warning(f"Image pair div not found on {url}")
        data['before_image_url'] = None
        data['after_image_url'] = None

    # Download images
    download_image(before_image_url, BEFORE_IMG_DIR, data['case_id'], session)
    download_image(after_image_url, AFTER_IMG_DIR, data['case_id'], session)

    # Extract patient info
    patient_info = scrape_patient_info(soup)
    data.update(patient_info) # Add patient details to main data dict

    return data # Return only the data

def load_existing_data(filepath):
    """Loads existing data from the JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {filepath}. Starting with empty list.")
            return []
        except IOError as e:
             logging.error(f"Error reading existing data file {filepath}: {e}. Starting with empty list.")
             return []
    return []

def save_data(data, filepath):
    """Saves data to the JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        # logging.info(f"Data successfully saved/updated to {filepath}") # Optional: reduce log noise
    except IOError as e:
        logging.error(f"Error writing to output file {filepath}: {e}")

def main(category_url, max_cases=75):
    """Main function to control the scraping process."""
    # Load existing data first
    all_data = load_existing_data(OUTPUT_JSON_FILE)
    existing_case_ids = {item['case_id'] for item in all_data if 'case_id' in item} # Set for faster lookups

    scraped_urls = set() # Keep track of visited URLs to prevent loops

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})

    # 1. Fetch the main category page
    logging.info(f"Fetching category page: {category_url}")
    try:
        response = session.get(category_url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching category URL {category_url}: {e}")
        return # Cannot proceed without category page

    # 2. Parse category page to find case URLs
    category_soup = BeautifulSoup(response.text, 'html.parser')
    case_links = []
    patient_items = category_soup.find_all('div', class_='patient-item')
    for item in patient_items:
        link_tag = item.find('a', href=True)
        if link_tag:
            case_url = urljoin(category_url, link_tag['href']) # Ensure absolute URL
            if case_url not in scraped_urls: # Avoid adding duplicates from category page itself
                 case_links.append(case_url)
                 scraped_urls.add(case_url) # Add here to avoid duplicates if multiple links point to same case

    if not case_links:
        logging.error(f"No case links found on category page {category_url}. Exiting.")
        return

    logging.info(f"Found {len(case_links)} unique case URLs.")

    # Limit to max_cases
    cases_to_scrape = case_links[:max_cases]
    logging.info(f"Processing the first {len(cases_to_scrape)} cases (or fewer if already scraped). Max set to {max_cases}.")

    new_cases_scraped = 0
    # 3. Iterate through case URLs and scrape
    for i, case_url in enumerate(cases_to_scrape):
        # Basic check if it looks like a valid case page URL (contains digits at the end)
        path_end = urlparse(case_url).path.strip('/').split('/')[-1]
        if not path_end.isdigit():
            logging.warning(f"Skipping URL, does not appear to be a case page: {case_url}")
            continue

        logging.info(f"Scraping case {i+1}/{len(cases_to_scrape)}: {case_url}")
        # Removed scraped_urls check here as we pre-filtered uniques

        page_data = scrape_case_page(case_url, session)

        if page_data and page_data.get('case_id'):
            if page_data['case_id'] not in existing_case_ids:
                all_data.append(page_data)
                existing_case_ids.add(page_data['case_id'])
                save_data(all_data, OUTPUT_JSON_FILE) # Save after adding new data
                logging.info(f"Saved data for case {page_data['case_id']}.")
                new_cases_scraped += 1
            else:
                 logging.info(f"Case {page_data['case_id']} already exists in JSON. Skipping append and save.")
        else:
             logging.warning(f"Failed to scrape valid data or case ID from {case_url}. Skipping.")

        time.sleep(0.5) # Reduced sleep time slightly

    logging.info(f"Finished scraping. Added {new_cases_scraped} new cases. Total cases in JSON: {len(all_data)}.")
    # Final save removed as it's done incrementally

if __name__ == "__main__":
    CATEGORY_URL = "https://skinperfectmedical.com/before-after-photos/face/"
    MAX_CASES_TO_SCRAPE = 75 # Define the limit here

    main(CATEGORY_URL, MAX_CASES_TO_SCRAPE) 