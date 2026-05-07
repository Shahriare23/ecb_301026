# ecb_301026
## Course name: Seminar on Media Economics


## Short summary on "Basic textual analysis on ECB Press Conference Workflow"

# My chosen ECB Press Conference 
I chose this press conference "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/html/ecb.is251030~4f74dde15e.en.html" for the given assignment because the impact of global trade and geopolitical tension on monetary policy piqued my intetrest. 

# Sentiment package used in this assignment:

I used VADER (Valence Aware Dictionary and sEntiment Reasoner) from NLTK in this assignment for the following reasons-
1. It's a beginner friendly package that allows the user to focus more on the actual analysis rather than debugging.
2. It was designed for modern communication text analysis.
3. It doesn't require any data model training and comes with pre-built thesaurus.
4. It has compound score range which easily identifies a wide range of sentiments (such as- very strong positive sentiment or mildly negative). 

# Result of the Sentiment Analysis:

a. Paragraph tone result:
The compound score 0.999 suggests a very confident and positive tone during the whole press conference. For the Q&A section, the socre 1.00 also indicates the same lacking any provocative response. 

b. Top 10 Words After Stopword Removal

| Rank | Word | Count | Theme |
|------|------|-------|-------|
| 1 | risks | 29 | Risk assessment |
| 2 | good | 27 | Optimistic framing |
| 3 | digital | 25 | Digital euro |
| 4 | risk | 24 | Risk assessment |
| 5 | banks | 23 | Banking stability |
| 6 | money | 21 | Digital euro / public good |
| 7 | growth | 20 | Economic outlook |
| 8 | financial | 19 | Banking stability |
| 9 | now | 19 | Current policy stance |
| 10 | public | 18 | Digital euro / public finances |

From the table above, we can confirm that, the key monetary policy topic "inflation" didn't make it in the top 10 postion avoiding any negative sentiment. Mr. Lagarde rather structured the speech in a positive frame focusing on the words above. They emphasized their efforts through "risk assesment", sovereignty and public welfare through "digital Euro" and the rest for a more optimismic frame. 