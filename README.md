# Literature wordcloud generator

Generate a word cloud image from article titles and abstracts, fetching data directly from **PubMed** and **Semantic Scholar**.

## Features

- Fetches papers from PubMed and Semantic Scholar APIs
- Supports filtering papers by required keywords
- Extracts terms using **TF-IDF**, **KeyBERT**, or a hybrid of both methods
- Saves fetched papers and computed term scores to JSON files
- Generates and saves a word cloud image (`wordcloud.png`)

## Requirements

- Python 3.12
- See `requirements.txt` for all dependencies:
  - `requests`
  - `nltk`
  - `scikit-learn`
  - `matplotlib`
  - `wordcloud`
  - `keybert`
  - `biopython`
  - `numpy`

## Usage

Run from the command line:

```bash
python word_cloud.py QUERY EMAIL [options]
```

### Positional arguments
QUERY : search query string

EMAIL : your email address (required for PubMed Entrez API)

### Optional arguments

-h, --help : show help message

-m, --method : keyword extraction method (tfidf, keybert, or hybrid; default: tfidf)

-n, --num : maximum number of articles per source (default: 20, maximum 100)

-f, --file : file path to save or load fetched papers (default: scraping_results.json)

-d, --debug : enable debug logging

-kw, --keywords : words that must be included in the title or the abstract


## Example
```bash
python word_cloud.py "laboratory automation" your_email@example.com -m hybrid -n 30 -kw automation lab testing
```
This will:
 - Fetch up to 30 papers per source containing “automation”, “lab”, or “testing”
 - Extract top keywords using the hybrid method
 - Generate and save a word cloud image as wordcloud.png

## Output
 - scraping_results.json : fetched papers data
 - term_scores.json : computed term scores used for word cloud
 - wordcloud.png : final word cloud image

