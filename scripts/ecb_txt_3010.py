from collections import Counter
from pathlib import Path
import re

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import STOPWORDS, WordCloud


# Step 1: We store the webpage address in a variable so we only need to write it once.
URL = "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/html/ecb.is251030~4f74dde15e.en.html"

# Step 2: We create folder paths for saved files.
# We make these now even though Step 1 only downloads the page.
# This keeps the project structure consistent from the beginning.
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

# Step 3: We ensure that the folders exist.
# exist_ok=True means Python will not crash if the folders are already there.
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def clean_whitespace(text: str) -> str:
    # Step 4: We turn messy spacing into clean spacing.
    """Turn repeated spaces, tabs, and newlines into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def sentiment_label(compound: float) -> str:
    # Step 5: We map the numeric compound score to a simple word label.
    """Convert a VADER compound score into a simple label."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def tokenize_words(text: str, stopwords: set[str]) -> list[str]:
    # Step 6: We break the text into lowercase word tokens.
    # We keep alphabetic words and allow apostrophes or hyphens inside them.
    # Then we remove short words and stopwords.
    """Make a simple list of lowercase words, excluding stopwords."""
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())
    return [
        token
        for token in tokens
        if len(token) > 2 and token not in stopwords
    ]

# Step 7: We send a browser-like identity to the website.
# Some websites are happier to respond when a request looks like it comes from a real browser.
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:150.0) Gecko/20100101 Firefox/150.0# (beginner text analysis tutorial)"
}

# Step 8: We download the ECB webpage.
# timeout=30 means Python will wait up to 30 seconds before giving up.
response = requests.get(URL, headers=headers, timeout=30)


# Step 9: We print simple checks so we know the download worked.
print("Downloaded page successfully")
print("Status code:", response.status_code)
print("Number of characters in HTML:", len(response.text))

# Step 10: We stop if the request failed.
response.raise_for_status()

# Step 11: We parse the HTML so we can find elements inside it.
soup = BeautifulSoup(response.text, "lxml")

# Step 12: We locate the main article section.
section = soup.select_one("main div.section")
if section is None:
    raise RuntimeError("Could not find the article section. The page structure may have changed.")

# Step 13: We remove elements that are not part of the real script text.
for unwanted in section.select('script, style, a[href="#qa"], .ecb-publicationDate'):
    unwanted.decompose()

# Step 14: We extract clean text from headings and paragraphs.
text_blocks = []
for element in section.find_all(["h2", "p"]):
    classes = element.get("class", [])

    # We skip the subtitle line with speaker names.
    if "ecb-pressContentSubtitle" in classes:
        continue

    # We pull out plain text from the HTML tag and clean the spacing.
    text = clean_whitespace(element.get_text(" ", strip=True))
    if text:
        text_blocks.append(text)

 # Step 14: We join everything into one full document.
full_text = "\n\n".join(text_blocks)

# Step 15: We save the extracted text as a plain text file.
text_path = DATA_DIR / "ecb_press_conference_2025-10-30.txt"
text_path.write_text(full_text, encoding="utf-8")

# Step 16: We download the VADER lexicon and build the sentiment analyzer.
nltk.download("vader_lexicon", quiet=True)
sentiment_analyzer = SentimentIntensityAnalyzer()

# Step 17: We score the full press conference as one document.
scores = sentiment_analyzer.polarity_scores(full_text)
scores["sentiment_label"] = sentiment_label(scores["compound"])
scores["source_url"] = URL

# Step 18: We save the sentiment result as a one-row CSV summary.
sentiment_summary = pd.DataFrame([scores])
sentiment_path = OUTPUT_DIR / "ecb_3010_sentiment_summary.csv"
sentiment_summary.to_csv(sentiment_path, index=False)

# Step 19: We start from the default English stopwords used by the wordcloud package.
custom_stopwords = set(STOPWORDS)

# Step 20: We add domain-specific words that would otherwise dominate the figure.
# These words are common in ECB texts, so removing them helps other themes stand out.
custom_stopwords.update(
    {
        "ecb",
        "euro",
        "area",
        "monetary",
        "policy",
        "inflation",
        "per",
        "cent",
        "will",
        "would",
        "could",
        "also",
        "question",
        "questions",
        "answer",
        "answers",
        "think",
        "going",
    }
)

# Step 21: We tokenize the clean text and remove stopwords.
tokens = tokenize_words(full_text, custom_stopwords)

# Step 22: We count how often each remaining word appears.
word_counts = Counter(tokens)

# Step 23: We save the top 30 words as a CSV table.
top_words = pd.DataFrame(
    word_counts.most_common(30),
    columns=["word", "count"],
)
top_words_path = OUTPUT_DIR / "ecb_3010_top_words.csv"
top_words.to_csv(top_words_path, index=False)

# Step 24: We build the word cloud image from the full text.
# The display settings control the size, colors, and reproducibility of the figure.
wordcloud = WordCloud(
    width=1200,
    height=700,
    background_color="white",
    stopwords=custom_stopwords,
    colormap="viridis",
    random_state=42,
).generate(full_text)

# Step 25: We choose the output path for the word cloud image.
wordcloud_path = OUTPUT_DIR / "ecb_3010_wordcloud.png"

# Step 26: We draw the word cloud with matplotlib and save it as a PNG file.
plt.figure(figsize=(12, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(wordcloud_path, dpi=200, bbox_inches="tight")
plt.close()

# Step 27: We print a short summary so we can confirm all output files were created.
print("Saved text to:", text_path)
print("Saved sentiment summary to:", sentiment_path)
print("Saved top words table to:", top_words_path)
print("Saved word cloud to:", wordcloud_path)
print()
print("Whole-document sentiment scores:")
print(sentiment_summary[["neg", "neu", "pos", "compound", "sentiment_label"]])
print()
print("Top 10 words after stopword removal:")
print(top_words.head(10))

# Step 28: We create another paragraph-level sentiment analysis
paragraph_data = []

for idx, paragraph in enumerate(text_blocks, start=1):
    # We get sentiment scores for this paragraph
    para_scores = sentiment_analyzer.polarity_scores(paragraph)
    
    # Store the information
    paragraph_data.append({
        "paragraph_number": idx,
        "paragraph_text": paragraph,
        "neg_score": para_scores["neg"],
        "neu_score": para_scores["neu"],
        "pos_score": para_scores["pos"],
        "compound_score": para_scores["compound"],
        "sentiment_label": sentiment_label(para_scores["compound"])
    })

# We convert to DataFrame and save
paragraph_sentiment_df = pd.DataFrame(paragraph_data)
paragraph_csv_path = OUTPUT_DIR / "ecb_3010_paragraph_sentiment.csv"
paragraph_sentiment_df.to_csv(paragraph_csv_path, index=False, encoding='utf-8')

print(f"Saved paragraph-level sentiment to: {paragraph_csv_path}")
print(f"Analyzed {len(paragraph_data)} paragraphs")

# Step 29: We separate Statement from Q&A


def separate_statement_from_qa(text_blocks_list: list[str]) -> tuple[list[str], list[str]]:
    """
    Split the press conference into two parts:
    - Statement: main speech before Q&A begins
    - Q&A: questions and answers after the anchor
    
    Returns: (statement_texts, qa_texts)
    """
    # We find where Q&A starts
    qa_start_index = -1
    
    for i, text in enumerate(text_blocks_list):
        # We look for the Q&A markers
        text_lower = text.lower()
        if ("questions and answers" in text_lower or 
            "we are now ready to take your questions" in text_lower or
            text.strip() == "* * *"):
            qa_start_index = i + 1  # Q&A starts after this element
            break
    
    # We split the text blocks
    if qa_start_index > 0 and qa_start_index < len(text_blocks_list):
        statement_texts = text_blocks_list[:qa_start_index]
        qa_texts = text_blocks_list[qa_start_index:]
    else:
        # If no Q&A marker found, put everything in statement
        statement_texts = text_blocks_list
        qa_texts = []
        print("Warning: Could not find Q&A separator. Using all text as statement.")
    
    return statement_texts, qa_texts


# We run the separation using your existing text_blocks
statement_blocks, qa_blocks = separate_statement_from_qa(text_blocks)

# We join into full strings
statement_full = "\n\n".join(statement_blocks)
qa_full = "\n\n".join(qa_blocks)

print(f"\n--- Separation Complete ---")
print(f"Statement: {len(statement_blocks)} elements, {len(statement_full):,} characters")
print(f"Q&A: {len(qa_blocks)} elements, {len(qa_full):,} characters")



# Step 29: Sentiment Analysis for statement and Q&A

# First we analyze statement
statement_scores = sentiment_analyzer.polarity_scores(statement_full)
statement_scores["sentiment_label"] = sentiment_label(statement_scores["compound"])
statement_scores["section"] = "statement"
statement_scores["source_url"] = URL

# We analyze Q&A 
if qa_blocks:
    qa_scores = sentiment_analyzer.polarity_scores(qa_full)
    qa_scores["sentiment_label"] = sentiment_label(qa_scores["compound"])
    qa_scores["section"] = "qa"
    qa_scores["source_url"] = URL
else:
    qa_scores = {"neg": 0, "neu": 0, "pos": 0, "compound": 0, 
                 "sentiment_label": "no_qa_found", "section": "qa", "source_url": URL}

# We save statement sentiment as individual CSV
statement_sentiment_df = pd.DataFrame([statement_scores])
statement_sentiment_path = OUTPUT_DIR / "ecb_statement_sentiment_summary.csv"
statement_sentiment_df.to_csv(statement_sentiment_path, index=False)

# We save Q&A sentiment as individual CSV
qa_sentiment_df = pd.DataFrame([qa_scores])
qa_sentiment_path = OUTPUT_DIR / "ecb_qa_sentiment_summary.csv"
qa_sentiment_df.to_csv(qa_sentiment_path, index=False)

print(f"\n--- Sentiment Results ---")
print(f"Saved statement sentiment to: {statement_sentiment_path}")
print(f"Saved Q&A sentiment to: {qa_sentiment_path}")
print(f"\nStatement: compound={statement_scores['compound']:.3f} ({statement_scores['sentiment_label']})")
print(f"Q&A: compound={qa_scores['compound']:.3f} ({qa_scores['sentiment_label']})")

#Step 30: We create separate word clouds for the Statement and Q&A sections.
def create_word_cloud_for_text(text: str, title: str, output_filename: str, output_dir: Path, stopwords_set: set):
    """Helper function to create and save a word cloud from text."""
    if not text or len(text.strip()) < 100:
        print(f"Skipping word cloud for {title} - text too short ({len(text)} characters)")
        return None
    
    # We create word cloud
    wc = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        stopwords=stopwords_set,
        colormap="viridis",
        random_state=42,
    ).generate(text)
    
    # We save the figure
    plt.figure(figsize=(12, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout(pad=0)
    
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"Saved word cloud for {title}: {output_path}")
    return output_path


# We create word cloud for Statement
if statement_full:
    create_word_cloud_for_text(
        statement_full,
        "ECB Monetary Policy Statement - October 30, 2025",
        "ecb_statement_wordcloud.png",
        OUTPUT_DIR,
        custom_stopwords
    )

# We create word cloud for Q&A
if qa_full:
    create_word_cloud_for_text(
        qa_full,
        "ECB Press Conference Q&A - October 30, 2025",
        "ecb_qa_wordcloud.png",
        OUTPUT_DIR,
        custom_stopwords
    )


# Step 30: We save statement and Q&A as text files


# We save the raw text for reference
statement_text_path = DATA_DIR / "ecb_statement_only_2025-10-30.txt"
statement_text_path.write_text(statement_full, encoding="utf-8")

qa_text_path = DATA_DIR / "ecb_qa_only_2025-10-30.txt"
if qa_full:
    qa_text_path.write_text(qa_full, encoding="utf-8")

print(f"\n--- Raw Text Files ---")
print(f"Saved statement text to: {statement_text_path}")
print(f"Saved Q&A text to: {qa_text_path}")



# Step 31: We print the final summary


print("\n" + "="*60)
print("FINAL OUTPUT SUMMARY")
print("="*60)
print("\n SENTIMENT FILES:")
print(f"   • {statement_sentiment_path}")
print(f"   • {qa_sentiment_path}")
print("\n WORD CLOUD FILES:")
print(f"   • {OUTPUT_DIR / 'ecb_statement_wordcloud.png'}")
print(f"   • {OUTPUT_DIR / 'ecb_qa_wordcloud.png'}")
print("\n DATA FILES (raw text):")
print(f"   • {statement_text_path}")
print(f"   • {qa_text_path}")
print("\n" + "="*60)

