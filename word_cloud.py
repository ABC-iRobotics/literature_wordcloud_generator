import time
import json
import logging
import itertools
import requests
import numpy as np
import matplotlib.pyplot as plt
from Bio import Entrez
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
from wordcloud import WordCloud

from nltk.stem import WordNetLemmatizer
import nltk


def fetch_pubmed_papers(num: int, query: str, must_include_words:list[str], email: str) -> list[dict[str,str]]:
    '''Fetch papers from PubMed

    Args
        num: Maximum number of papers to retrieve
        query: Search query
        must_include_words: List of words that must be included in the title or the abstract
        email: Email address provided to Entrez
    
    Returns
        Papers as a list of dicts. Each dict is a paper containing the paper source (pubmed), id, title, abstract and year 
    '''
    global logger

    # Search PubMed DB (look for query in title and abstract)
    Entrez.email = email
    handle = Entrez.esearch(db="pubmed", term=query, retmax=num, field="title/abstract")
    record = Entrez.read(handle)
    ids = record["IdList"]

    #Fetch abstracts
    papers = []
    for pmid in ids:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
        records = Entrez.read(handle)

        # Parse the XML
        article = records['PubmedArticle'][0]
        title = article['MedlineCitation']['Article']['ArticleTitle']
        pubdate = article['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']
        year = pubdate.get('Year') or pubdate.get('MedlineDate', '').split(' ')[0]

        # Abstract may have multiple sections; join them
        abstract = ''
        if 'Abstract' in article['MedlineCitation']['Article']:
            abstract_sections = article['MedlineCitation']['Article']['Abstract']['AbstractText']
            abstract = ' '.join(abstract_sections)
        
        if abstract:
            papers.append({
                'source': 'pubmed',
                'id': pmid,
                'title': title,
                'abstract': abstract,
                'year': year
            })
            logger.info(title)

        # Sleep to avoid IP blocking due-to frequent requests
        time.sleep(0.34)

    # Filter papers that contain the provided "must_include_words"
    if must_include_words:
        relevant_papers = []
        for paper in papers:
            if any(kw.lower() in paper['title'].lower() or kw.lower() in paper.get('abstract', '').lower() for kw in must_include_words):
                relevant_papers.append(paper)
        return relevant_papers
    else:
        return papers


def fetch_semantic_scholar_papers(num: int, query: str, must_include_words: list[str]) -> list[dict[str,str]]:
    '''Fetch papers from SemanticScholar

    Args
        num: Maximum number of papers to retrieve
        query: Search query
        must_include_words: List of words that must be included in the title or the abstract
    
    Returns
        Papers as a list of dicts. Each dict is a paper containing the paper source (semanticscholar), id, title, abstract and year 
    '''
    global logger
    # Construct URL from query
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={num}&fields=title,abstract,year,authors"

    # Fetch data
    response = requests.get(url)
    data = response.json()

    papers = []
    for paper in data.get('data',[]):
        if paper.get('abstract', ''):
            papers.append({
                'source': 'semanticscholar',
                'id': paper['paperId'],
                'title': paper['title'],
                'abstract': paper.get('abstract', ''),
                'year': paper.get('year')
            })
            logger.info(paper['title'])

    if must_include_words:
        relevant_papers = []
        for paper in papers:
            if any(kw.lower() in paper['title'].lower() or kw.lower() in paper.get('abstract', '').lower() for kw in must_include_words):
                relevant_papers.append(paper)
        return relevant_papers
    else:
        return papers




if __name__ == '__main__':
    import argparse
    import socket

    ####################################################################
    # Initial setup

    # Arguments
    parser = argparse.ArgumentParser(
        description='Create word cloud from queries'
    )
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('email', type=str, help='Email address provided to Entrez')
    parser.add_argument('-m', '--method', type=str, default='tfidf', choices=['tfidf', 'keybert', 'hybrid'])
    parser.add_argument('-n', '--num', type=int, default=20, help='Maximum number of articles per source')
    parser.add_argument('-f', '--file', type=str, default='scraping_results.json', help='Papers file path')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-kw', '--keywords', type=str, nargs='+', help='Words that must be included in the title or the abstract')
    args = parser.parse_args()

    # Logging
    global logger
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(
        __name__ + '.' + socket.gethostname()
    )
    loglevel = logging.INFO if args.debug else logging.WARNING
    logger.setLevel(logging.INFO)


    ####################################################################
    # Get papers

    # Build queries
    pubmed_query = (f'("{args.query}"[Title/Abstract])')
    semanticscholar_query = args.query

    try:
        # Load from file if it exists
        logger.info('Attempting to load papers from file')
        with open(args.file, 'r') as f:
            relevant_papers = json.loads(f.read())
    except FileNotFoundError:
        # Request papers if there is no file found
        logger.info(
            f'No saved papers found at {args.file}, fetching them from PubMed and SemanticScholar'
        )
        relevant_papers_pubmed = fetch_pubmed_papers(args.num, pubmed_query, args.keywords, args.email)
        relevant_papers_sscholar = fetch_semantic_scholar_papers(args.num, semanticscholar_query, args.keywords)

        logger.info('Finished fetching papers')

        # Remove duplicates based on title
        unique_paper_titles = list({p['title'] for p in itertools.chain(relevant_papers_pubmed, relevant_papers_sscholar)})
        relevant_papers = [p for p in itertools.chain(relevant_papers_pubmed, relevant_papers_sscholar) if p['title'] in unique_paper_titles]

    # Save papers
    with open(args.file,'w') as f:
        logger.info('Saving papers to file')
        f.write(json.dumps(relevant_papers))


    ####################################################################
    # Extract frequent terms

    # Use TF-IDF to find frequent terms
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([p['abstract'] for p in relevant_papers])

    terms = vectorizer.get_feature_names_out()
    scores = [float(v) for v in np.asarray(tfidf_matrix.sum(axis=0)).ravel()]

    tfidf_keywords = dict(zip(terms, scores))
    print('TFIDF terms: ', tfidf_keywords)

    # Use KeyBERT for extracting contextually significant terms
    kw_model = KeyBERT()
    keybert_keywords = {}
    for abstract in [p['abstract'] for p in relevant_papers]:
        keywords = kw_model.extract_keywords(
            abstract,
            stop_words='english',
            top_n=5)
        for kw, score in keywords:
            if keybert_keywords.get(kw.lower(), None):
                keybert_keywords[kw.lower()] += score
            else:
                keybert_keywords[kw.lower()] = score

    print('KeyBERT keywords: ', keybert_keywords)

    if args.method == 'keybert':
        # Sum KeyBERT scores
        terms = [t for t in keybert_keywords]
        scores = [v for k,v in keybert_keywords.items()]

    if args.method == 'hybrid':
        # Multiply TF-IDF and KeyBERT scores
        terms = []
        scores = []
        for term,score in keybert_keywords.items():
            tfidf_score = tfidf_keywords.get(term, None)
            if tfidf_score:
                score = tfidf_score*score
                terms.append(term)
                scores.append(score)

    # Use Wordnet for lemmatize terms (remove duplicates caused by plural form etc.)
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

    combined_tfidf = {}
    for word, score in zip(terms, scores):
        lemma = lemmatizer.lemmatize(word.lower(), pos='n')
        if lemma in combined_tfidf:
            combined_tfidf[lemma] = max(combined_tfidf[lemma], score)  # Keep lemma and associate the higher score
        else:
            combined_tfidf[lemma] = score

    # Filter out terms that are not in KeyBERT keywords and those that are in args.keywords
    filtered_terms = []
    filtered_scores = []
    for term, score in combined_tfidf.items():
        for kw in keybert_keywords:
            if term in kw.split() and term not in args.keywords and term not in ['automating', 'automate', 'development', 'test', 'testing']:
                filtered_terms.append(term)
                filtered_scores.append(score)
                break  # stop checking other keywords if term is found


    # Rank by TF-IDF scores
    filtered_scores = np.array(filtered_scores)
    top_indices = [i for i in filtered_scores.argsort()[-50:][::-1]]
    top_terms = [filtered_terms[i] for i in top_indices]
    top_scores = [float(filtered_scores[i]) for i in top_indices]
    term_scores = dict(zip(top_terms, top_scores))
    print("Top TF-IDF terms:", term_scores)

    with open('term_scores.json','w') as f:
        logger.info('Saving terms and scores to file')
        f.write(json.dumps(term_scores))


    ####################################################################
    # Visualize

    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(term_scores)

    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("wordcloud.png", dpi=300, bbox_inches='tight')

