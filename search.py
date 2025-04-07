#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import random
import time
import sys
import argparse
import json
import urllib.parse
import os
import hashlib
import logging
from datetime import datetime, timedelta
import re
from typing import List, Dict, Union, Optional, Tuple
from newspaper import Article
from readability import Document

# --- Settings ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:12b"
MAX_SEARCH_RESULTS = 3
MAX_CHARS_PER_PAGE = 4000  # Increased as we have better extraction methods
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
USE_CACHE = True
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CACHE_EXPIRATION = timedelta(hours=24)

# Create cache directory if it doesn't exist
if USE_CACHE and not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("web_research.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("web_research")

# Force UTF-8 for output
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# List of User-Agents for requests
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/97.0.1072.55",
    "Mozilla/5.0 (iPad; CPU OS 15_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:97.0) Gecko/20100101 Firefox/97.0"
]

# --- DuckDuckGo search functions ---

def get_cache_path(query: str) -> str:
    """
    Gets the path to the cache file for a query.
    
    Args:
        query (str): Search query
        
    Returns:
        str: Path to the cache file
    """
    # Create a hash of the query to use as the filename
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{query_hash}.json")

def get_cached_results(query: str) -> Optional[List[Dict]]:
    """
    Gets cached results for a query if they exist and aren't outdated.
    
    Args:
        query (str): Search query
        
    Returns:
        list or None: Cached results or None if not found or outdated
    """
    if not USE_CACHE:
        return None
        
    cache_path = get_cache_path(query)
    
    if not os.path.exists(cache_path):
        return None
        
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
            
        # Check if the cache is outdated
        cached_time = datetime.fromisoformat(cached_data['timestamp'])
        if datetime.now() - cached_time > CACHE_EXPIRATION:
            logger.debug(f"Cache for query '{query}' is outdated.")
            return None
            
        logger.info(f"Using cached results for '{query}'")
        return cached_data['results']
    except Exception as e:
        logger.warning(f"Error reading cache: {e}")
        return None

def save_to_cache(query: str, results: List[Dict]) -> None:
    """
    Saves search results to cache.
    
    Args:
        query (str): Search query
        results (list): Search results to cache
    """
    if not USE_CACHE or not results:
        return
        
    cache_path = get_cache_path(query)
    
    try:
        cache_data = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
        logger.debug(f"Results for '{query}' saved to cache")
    except Exception as e:
        logger.warning(f"Error saving to cache: {e}")

def get_random_user_agent() -> str:
    """Returns a random User-Agent from the list of popular browsers."""
    return random.choice(USER_AGENTS)

def _make_request(url: str) -> Optional[requests.Response]:
    """
    Makes an HTTP request with retries and exponential backoff.
    
    Args:
        url (str): URL to request
        
    Returns:
        Response or None: Response object or None if the request failed
    """
    retry_count = 0
    
    while retry_count < MAX_RETRIES:
        try:
            # Add random delay to simulate human behavior
            time.sleep(random.uniform(0.5, 2.0))
            
            # Get a random user agent and set up headers
            user_agent = get_random_user_agent()
            headers = {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://duckduckgo.com/",
                "DNT": "1",  # Do Not Track
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "same-origin"
            }
            
            # Make the request
            response = requests.get(
                url,
                headers=headers,
                cookies={"ax": str(random.randint(1, 9))},
                timeout=REQUEST_TIMEOUT
            )
            
            # Check for success
            if response.status_code == 200:
                # Check for CAPTCHA or blocking
                if any(term in response.text.lower() for term in ["captcha", "blocked", "too many requests"]):
                    logger.warning(f"CAPTCHA or blocking detected for {url}. Aborting this URL.")
                    # Don't retry the same URL if blocked, just return None
                    return None
                    
                return response
                
            elif response.status_code == 429 or response.status_code >= 500:
                # Too many requests or server error - RETRY
                logger.warning(f"Received status code {response.status_code} for {url}. Retrying...")
                retry_count += 1
                time.sleep(2 ** retry_count + random.uniform(1, 3))
            else:
                logger.error(f"Error: Received status code {response.status_code}")
                
                # Debug: save the response for inspection
                debug_path = f"debug_response_{response.status_code}.html"
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                logger.debug(f"Error response saved to {debug_path}")
                
                return None
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error: {e}")
            retry_count += 1
            time.sleep(2 ** retry_count + random.uniform(1, 3))
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
            
    logger.error(f"Failed to make request after {MAX_RETRIES} attempts")
    return None

def _find_element(parent_element, selectors: List[str]):
    """
    Tries multiple selectors to find an element.
    
    Args:
        parent_element: Parent BeautifulSoup element
        selectors (list): List of CSS selectors to try
        
    Returns:
        Found element or None
    """
    for selector in selectors:
        found = parent_element.select_one(selector)
        if found:
            return found
    
    # If no selector worked, try direct search of common elements
    if selectors[0].endswith('a'):  # If searching for links
        return parent_element.find('a')
    elif any(s.endswith('snippet') for s in selectors):  # If searching for text/snippet
        # Try to find paragraph or div without links
        for tag in parent_element.find_all(['p', 'div', 'span']):
            if not tag.find('a') and tag.get_text().strip():
                return tag
    
    return None

def _find_results_alternative(soup: BeautifulSoup) -> List:
    """
    Alternative method of finding results when standard selectors don't work.
    
    Args:
        soup (BeautifulSoup): Parsed HTML
        
    Returns:
        list: Found result elements
    """
    results = []
    
    # Strategy 1: Look for links in headings
    heading_links = soup.select('h2 a, h3 a, h4 a')
    if heading_links:
        logger.debug(f"Found {len(heading_links)} potential results using heading links")
        # For each link in a heading, find its containing block (likely a result container)
        for link in heading_links:
            # Walk up a few levels to find potential container
            container = link
            for _ in range(3):  # Try to walk up to 3 levels
                if container.parent:
                    container = container.parent
                else:
                    break
            
            # Add only containers we haven't seen yet
            if container not in results:
                results.append(container)
    
    # Strategy 2: Look for clusters of links
    if len(results) < 5:  # If we didn't find enough results
        logger.debug("Looking for link clusters as a fallback")
        links = soup.find_all('a')
        
        # Group links by their parents
        link_parents = {}
        for link in links:
            parent = link.parent
            if parent:
                if parent not in link_parents:
                    link_parents[parent] = []
                link_parents[parent].append(link)
        
        # Look for parents with exactly one link (potential results)
        for parent, parent_links in link_parents.items():
            if len(parent_links) == 1 and parent not in results:
                # Walk up one level to get the container
                container = parent.parent if parent.parent else parent
                if container not in results:
                    results.append(container)
    
    return results

def _extract_html_results(html_content: str) -> List[Dict]:
    """
    Extracts search results from DuckDuckGo HTML version.
    
    Args:
        html_content (str): HTML content
        
    Returns:
        list: Extracted search results
    """
    results = []
    
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Save the content for debugging if verbose output is enabled
        if logger.level <= logging.DEBUG:
            with open("debug_html_content.html", "w", encoding="utf-8") as f:
                f.write(html_content)
        
        # Try several selector patterns to find results
        selectors_to_try = [
            # Standard results
            "div.result",
            "div.results_links",
            "div.results_links_deep",
            "div.web-result",
            # Newer format results
            "article.result",
            "article.result__web",
            "div.PartialWebResult",
            "div.PartialWebResult-title",
            # Additional modern selectors
            "div[data-testid='result']",
            "div[data-testid='web-result']",
            "div.zci__result"
        ]
        
        result_elements = []
        for selector in selectors_to_try:
            result_elements = soup.select(selector)
            if result_elements:
                logger.debug(f"Found {len(result_elements)} results using selector: {selector}")
                break
        
        # If results not found using selectors, try alternative method
        if not result_elements or len(result_elements) < 3:  # Less than 3 results is suspicious
            logger.debug("Standard selectors didn't find enough results, trying alternative method")
            result_elements = _find_results_alternative(soup)
        
        # Process found elements
        for result in result_elements:
            try:
                # Look for title, link and description using multiple potential selectors
                title_element = _find_element(result, [
                    "a.result__a", 
                    "a.result__url", 
                    "a[data-testid='result-title-a']",
                    "a.title", 
                    "h2 a", 
                    "h3 a",
                    ".result__title a",
                    ".PartialWebResult-title a"
                ])
                
                if not title_element:
                    continue
                    
                title = title_element.get_text().strip()
                link = title_element.get("href", "")
                
                # Skip internal links like "javascript:" or None
                if not link or link.startswith("javascript:") or link == "#":
                    continue
                    
                # Process DuckDuckGo redirect URLs
                if "/y.js?" in link or "/l.js?" in link or "uddg=" in link:
                    # Extract the actual URL from DuckDuckGo redirect
                    parsed_url = urllib.parse.urlparse(link)
                    query_params = urllib.parse.parse_qs(parsed_url.query)
                    
                    if 'uddg' in query_params:
                        link = urllib.parse.unquote(query_params['uddg'][0])
                    elif 'u' in query_params:
                        link = urllib.parse.unquote(query_params['u'][0])
                    else:
                        # If we can't extract the URL, skip
                        continue
                
                # Look for description
                desc_element = _find_element(result, [
                    "a.result__snippet", 
                    "div.result__snippet", 
                    ".snippet", 
                    ".snippet__content",
                    "div[data-testid='result-snippet']",
                    ".PartialWebResult-snippet",
                    ".result__body"
                ])
                
                description = ""
                if desc_element:
                    description = desc_element.get_text().strip()
                
                # Add to results if we have both a title and a link
                if title and link:
                    results.append({
                        "title": title,
                        "link": link,
                        "description": description
                    })
            except Exception as e:
                logger.debug(f"Error processing result element: {e}")
                continue
    except Exception as e:
        logger.error(f"Error extracting HTML results: {e}")
    
    return results

def _extract_lite_results(html_content: str) -> List[Dict]:
    """
    Extracts search results from DuckDuckGo lite version.
    
    Args:
        html_content (str): HTML content
        
    Returns:
        list: Extracted search results
    """
    results = []
    
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Lite version uses simple table markup
        # Look for table rows with links
        result_rows = soup.select('table tr:has(a)')
        
        # If we didn't find results using the selector, try a more general approach
        if not result_rows:
            logger.debug("Standard lite selectors didn't work, trying alternative approach")
            result_rows = []
            for table in soup.find_all('table'):
                for row in table.find_all('tr'):
                    if row.find('a'):
                        result_rows.append(row)
        
        for row in result_rows:
            try:
                # Find the link element
                link_elem = row.find('a')
                if not link_elem:
                    continue
                
                title = link_elem.get_text().strip()
                link = link_elem.get('href', '')
                
                # Skip empty links or internal links
                if not link or link.startswith('/'):
                    continue
                
                # In the lite version, the description is often in the next row
                description = ""
                next_row = row.find_next_sibling('tr')
                if next_row:
                    # Description is usually in the first cell without a link
                    desc_cells = [cell for cell in next_row.find_all('td') if not cell.find('a')]
                    if desc_cells:
                        description = desc_cells[0].get_text().strip()
                
                # Add to results
                if title and link:
                    results.append({
                        "title": title,
                        "link": link,
                        "description": description
                    })
            except Exception as e:
                logger.debug(f"Error processing lite result row: {e}")
                continue
    except Exception as e:
        logger.error(f"Error extracting lite results: {e}")
    
    return results

def _search_html_version(query: str) -> List[Dict]:
    """
    Search using DuckDuckGo HTML version.
    
    Args:
        query (str): Search query
        
    Returns:
        list: Search results
    """
    encoded_query = urllib.parse.quote(query)
    url = f"https://duckduckgo.com/html/?q={encoded_query}"
    
    # Add some random parameters to look more like a browser
    params = []
    possible_params = [
        ("kl", random.choice(["ru-ru", "us-en", "wt-wt"])),  # Locale
        ("k1", "-1"),  # Disable safe search
        ("kp", str(random.choice([-2, -1, 0, 1, 2]))),  # Filter level
        ("kaf", str(random.randint(0, 1))),  # Auto correction
        ("kd", str(random.randint(0, 1))),  # Suggestions
    ]
    
    selected_params = random.sample(possible_params, k=random.randint(1, len(possible_params)))
    params = [f"{k}={v}" for k, v in selected_params]
    
    if params:
        url = f"{url}&{'&'.join(params)}"
        
    response = _make_request(url)
    if not response:
        return []
        
    return _extract_html_results(response.text)

def _search_lite_version(query: str) -> List[Dict]:
    """
    Search using DuckDuckGo lite version.
    
    Args:
        query (str): Search query
        
    Returns:
        list: Search results
    """
    encoded_query = urllib.parse.quote(query)
    url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}"
    
    response = _make_request(url)
    if not response:
        return []
        
    return _extract_lite_results(response.text)

def search_duckduckgo(query: str) -> List[Dict]:
    """
    Search DuckDuckGo for a given query.
    
    Args:
        query (str): Search query
        
    Returns:
        list: Search results
    """
    # First check the cache
    cached_results = get_cached_results(query)
    if cached_results is not None:
        return cached_results
        
    logger.info(f"Searching DuckDuckGo for: {query}")
    
    # Try both HTML version and lite version
    results = _search_html_version(query)
    
    if not results:
        logger.info("HTML version failed, trying lite version")
        results = _search_lite_version(query)
        
    # If we got results, cache them
    if results:
        save_to_cache(query, results)
        
    return results

def google_fallback_search(query: str, num_results=MAX_SEARCH_RESULTS) -> List[Dict]:
    """
    Google search as a fallback if DuckDuckGo doesn't work.
    
    Args:
        query (str): Search query
        num_results (int): Maximum number of results
        
    Returns:
        list: Search results
    """
    logger.info(f"Using Google as a fallback for search: '{query}'...")
    results = []
    try:
        # Form the Google search URL
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://www.google.com/search?q={encoded_query}&hl=ru&num={num_results*2}"
        
        response = _make_request(search_url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find results using different selectors
        search_results = []
        
        # Modern Google selectors
        for div in soup.select("div.g"):
            item = {}
            a_tag = div.select_one("a")
            if a_tag and 'href' in a_tag.attrs:
                item['link'] = a_tag['href']
                h3 = div.select_one("h3")
                if h3:
                    item['title'] = h3.get_text()
                    # Look for description
                    desc_div = div.select_one("div.VwiC3b") or div.select_one("span.aCOpRe")
                    if desc_div:
                        item['description'] = desc_div.get_text()
                    else:
                        item['description'] = ""
                    
                    search_results.append(item)
        
        # If we didn't find results using modern selectors, try older ones
        if not search_results:
            for result in soup.select(".tF2Cxc"):
                item = {}
                a_tag = result.select_one("a")
                if a_tag and 'href' in a_tag.attrs:
                    item['link'] = a_tag['href']
                    h3 = result.select_one("h3") or result.select_one(".DKV0Md")
                    if h3:
                        item['title'] = h3.get_text()
                        # Look for description
                        desc = result.select_one(".IsZvec") or result.select_one(".lEBKkf")
                        if desc:
                            item['description'] = desc.get_text()
                        else:
                            item['description'] = ""
                        
                        search_results.append(item)
        
        # Filter and limit results
        filtered_results = []
        for result in search_results:
            if 'link' in result and not result['link'].startswith("/"):
                if 'title' in result and len(filtered_results) < num_results:
                    filtered_results.append(result)
        
        return filtered_results
    except Exception as e:
        logger.error(f"Error in Google fallback search: {e}")
        return []

def search_web_with_retries(query: str, num_results=MAX_SEARCH_RESULTS, max_retries=MAX_RETRIES) -> List[Dict]:
    """
    Searches for query in DuckDuckGo with retry mechanism and fallbacks.
    
    Args:
        query (str): Search query
        num_results (int): Maximum number of results
        max_retries (int): Maximum number of retries
        
    Returns:
        list: URLs of found results
    """
    print(f"[*] Searching DuckDuckGo for: '{query}'...")
    
    # Try DuckDuckGo search with advanced module
    results = search_duckduckgo(query)
    
    # If we found results, return URLs
    if results:
        urls = [r['link'] for r in results[:num_results]]
        print(f"[*] Found {len(urls)} links.")
        return urls
    
    # If DuckDuckGo gave no results, use Google
    print("[!] DuckDuckGo returned no results. Trying Google...")
    google_results = google_fallback_search(query, num_results)
    
    if google_results:
        urls = [r['link'] for r in google_results]
        print(f"[*] Google found {len(urls)} links.")
        return urls
    
    # If Google gave no results either, use predefined URLs for bitcoin queries
    if not google_results and ("биткоин" in query.lower() or "bitcoin" in query.lower()):
        default_urls = [
            "https://www.coindesk.com/price/bitcoin/",
            "https://ru.investing.com/crypto/bitcoin",
            "https://www.blockchain.com/explorer/assets/btc",
            "https://bitinfocharts.com/ru/bitcoin/"
        ]
        print("[*] Using predefined list of cryptocurrency websites.")
        return default_urls[:num_results]
    
    # If nothing helped
    print("[!] Could not find results from any source.")
    return []

# --- Functions for extracting text from web pages ---

def clean_text(text):
    """Additional cleaning of text from garbage."""
    # Remove extra spaces and line breaks
    text = re.sub(r'\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove common advertisement and social media elements
    patterns_to_remove = [
        r'Subscribe to.*',
        r'Read also:.*',
        r'Share:.*',
        r'Share.*',
        r'Comments.*',
        r'Copyright ©.*',
        r'\d+ comment(s)?.*',
        r'Advertisement.*',
        r'Advertisement.*',
        r'Loading comments.*',
        r'Popular:.*',
        r'Related:.*',
        r'Source:.*',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()

def method1_bs4(url):
    """Method 1: Parsing with BeautifulSoup."""
    try:
        response = _make_request(url)
        if not response:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        # Find main content
        article_content = ""
        
        # First look for article tag
        article_tag = soup.find('article')
        if article_tag:
            for p in article_tag.find_all('p'):
                article_content += p.get_text() + "\n\n"
            if article_content:
                return article_content.strip()
        
        # Look for divs with classes containing 'content' or 'article'
        content_divs = soup.find_all('div', class_=lambda c: c and ('content' in c.lower() or 'article' in c.lower()))
        for div in content_divs:
            for p in div.find_all('p'):
                article_content += p.get_text() + "\n\n"
            if article_content:
                return article_content.strip()
        
        # Just collect all paragraphs
        if not article_content:
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                article_content += p.get_text() + "\n\n"
        
        return article_content.strip()
    except Exception as e:
        logger.error(f"Error in method 1 (BS4): {str(e)}")
        return None

def method2_newspaper(url):
    """Method 2: Parsing with Newspaper3k library."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        if article.text:
            return article.text.strip()
        else:
            return None
    except Exception as e:
        logger.error(f"Error in method 2 (Newspaper): {str(e)}")
        return None

def method3_readability(url):
    """Method 3: Parsing with Readability library."""
    try:
        response = _make_request(url)
        if not response:
            return None
            
        doc = Document(response.text)
        content = doc.summary()
        
        # Clean from HTML tags
        soup = BeautifulSoup(content, 'html.parser')
        clean_text_content = soup.get_text()
        
        # Remove extra spaces and line breaks
        clean_text_content = re.sub(r'\n+', '\n\n', clean_text_content)
        clean_text_content = re.sub(r' +', ' ', clean_text_content)
        
        return clean_text_content.strip()
    except Exception as e:
        logger.error(f"Error in method 3 (Readability): {str(e)}")
        return None

def method4_direct_extraction(url):
    """Method 4: Direct text extraction from any elements."""
    try:
        response = _make_request(url)
        if not response:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'aside']):
            tag.decompose()
        
        # Collect text from all elements
        all_text = soup.get_text(separator='\n')
        
        # Clean text
        cleaned_text = re.sub(r'\n+', '\n\n', all_text)
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        
        return cleaned_text.strip()
    except Exception as e:
        logger.error(f"Error in method 4 (Direct extraction): {str(e)}")
        return None

def fetch_and_extract_text(url):
    """
    Loads a URL and extracts the main text using multiple methods.
    
    Args:
        url (str): URL to load and extract
        
    Returns:
        str or None: Extracted text or None in case of error
    """
    print(f"[*] Loading and extracting text from: {url}")
    
    # Try several text extraction methods and choose the best result
    methods = [
        ("Readability", method3_readability(url)),
        ("Newspaper", method2_newspaper(url)),
        ("BeautifulSoup", method1_bs4(url)),
        ("Direct extraction", method4_direct_extraction(url))
    ]
    
    # Filter results, excluding None and too short texts
    valid_results = [(name, text) for name, text in methods if text and len(text) > 150]
    
    if not valid_results:
        print("[!] Failed to extract text with any method.")
        return None
    
    # Choose the longest text
    best_method, best_text = max(valid_results, key=lambda x: len(x[1]))
    
    # Additional text cleaning
    cleaned_text = clean_text(best_text)
    
    print(f"[*] Extracted ~{len(cleaned_text)} characters using {best_method} method.")
    return cleaned_text[:MAX_CHARS_PER_PAGE]  # Limit text length

# --- Functions for working with Ollama ---

def query_ollama(prompt, model=MODEL_NAME):
    """
    Sends a prompt to the Ollama model and returns the response.
    
    Args:
        prompt (str): Prompt text
        model (str): Model name
        
    Returns:
        str or None: Model response or None in case of error
    """
    logger.info(f"Sending request to model {model}...")
    
    for attempt in range(MAX_RETRIES):
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False  # Get the response as a whole, not in parts
            }
            
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
            response.raise_for_status()

            response_data = response.json()
            if 'response' in response_data:
                logger.info("Response from LLM received.")
                return response_data['response']
            else:
                logger.warning(f"Unexpected response format from Ollama: {response_data}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt+1}/{MAX_RETRIES}: Error when contacting Ollama API: {e}")
            if attempt < MAX_RETRIES - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Waiting {wait_time:.2f} seconds before next attempt...")
                time.sleep(wait_time)
            else:
                return None
        except json.JSONDecodeError:
            logger.error("Error decoding JSON response from Ollama.")
            return None
        except Exception as e:
            logger.error(f"Unknown error working with Ollama: {e}")
            return None
    
    # If Ollama is not responding, provide basic information
    logger.error("Could not get a response from LLM after multiple attempts.")
    return "Could not get a response from LLM. Please check if the Ollama API is running and the model is available."

# --- Core Deep Search Function ---

def perform_deep_search(user_query: str) -> str:
    """
    Performs the complete deep search process for a given query.
    Searches the web, extracts text, and queries Ollama for a synthesized answer.

    Args:
        user_query (str): The user's research query.

    Returns:
        str: The synthesized answer from the LLM, or an error message.
    """
    try:
        logger.info(f"Performing deep search for: {user_query}")
        urls_to_process = search_web_with_retries(user_query)

        if not urls_to_process:
            logger.warning("Deep search failed to find sources.")
            # Form prompt for LLM without context from web sources
            backup_prompt = f"""
            Please provide information on the following query using your base knowledge: 
            "{user_query}"
            
            Important: since we were unable to retrieve current data from web sources, your answer may not include 
            the most recent information. Indicate this limitation in your response.
            """
            llm_response = query_ollama(backup_prompt)
            return llm_response if llm_response else "Could not find sources and failed to get a base response from LLM."

        logger.info(f"Processing {len(urls_to_process)} found links for deep search...")
        all_extracted_text = ""
        processed_count = 0
        for url in urls_to_process:
            extracted_text = fetch_and_extract_text(url)
            if extracted_text:
                all_extracted_text += f"\n\n--- Source: {url} ---\n{extracted_text}"
                processed_count += 1

        if not all_extracted_text:
            logger.warning("Failed to extract text from any of the sources for deep search.")
            # Form prompt for LLM without context from web sources
            backup_prompt = f"""
            Please provide information on the following query using your base knowledge: 
            "{user_query}"
            
            Important: since we were unable to retrieve current data from web sources, your answer may not include 
            the most recent information. Indicate this limitation in your response.
            """
            llm_response = query_ollama(backup_prompt)
            return llm_response if llm_response else "Found sources but failed to extract text and failed to get a base response from LLM."

        logger.info(f"Collected {len(all_extracted_text)} characters of text from {processed_count} sources for deep search.")

        # Form prompt for LLM
        final_prompt = f"""
        Based on the text below, collected from various web sources, answer the following research question: "{user_query}"

        Please provide a structured and informative answer, synthesizing information from the texts. Highlight key points. If information in the sources is contradictory, note this. Do not add information that is not in the provided texts.

        Here is the text from the sources:
        {all_extracted_text}

        Answer to the question "{user_query}":
        """

        llm_response = query_ollama(final_prompt)
        return llm_response if llm_response else "Successfully gathered web context but failed to get a final response from LLM."

    except Exception as e:
        logger.error(f"An unexpected error occurred during deep search: {e}")
        logger.error(traceback.format_exc())
        return f"An error occurred during the deep search process: {str(e)}"

# --- Main code (for standalone execution) ---

def main():
    try:
        user_query = input("Enter your research query: ")
        result = perform_deep_search(user_query)
        # Use logger instead of print for main execution as well
        logger.info("\n\n✅ Research result:")
        logger.info("=" * 25)
        logger.info(result)
        logger.info("=" * 25)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}")
        import traceback
        logger.error(traceback.format_exc()) # Print stack trace for detailed debugging
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred at top level: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)