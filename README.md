# Drug Review Analysis

A comprehensive analysis of drug reviews leveraging statistical methods, Natural Language Processing (NLP), and Large Language Models (LLMs). This project explores sentiment analysis, text summarization, and insights generation to evaluate drug efficacy and user sentiment.

---

## Features

- **Sentiment Analysis**:
  - BERT-based sentiment analysis optimized using batch processing on GPU for efficiency.
  - Sentiment analysis using Gemini API, OpenAI API, and GROQ API with performance comparisons.
  - Batch processing optimization for Gemini API to handle up to 100 reviews in a single API call, overcoming free-tier limitations.

- **Text Summarization**:
  - Summarized drug reviews for each drug using Gemini API to provide aggregated insights.

- **Data Insights**:
  - Statistical analysis of drug reviews to identify key patterns and trends.

---
## Notebook Interaction

These notebooks can be viewed using [nbviewer](https://nbviewer.org/), which provides an enhanced, interactive experience for exploring the data and analysis:

- [Drug Review Overall Analysis](https://nbviewer.org/github/yangsong24/drug-review-analysis/blob/main/Drug_review.ipynb)
- [Sentiment Analysis](https://nbviewer.org/github/yangsong24/drug-review-analysis/blob/main/sentiment_analysis.ipynb)
- [Batch-Processing Optimization Code](https://nbviewer.org/github/yangsong24/drug-review-analysis/blob/main/BP_review_analysis.ipynb)

---

## Dataset

The dataset contains user reviews for various drugs, including feedback on their effectiveness, side effects, and conditions they treat. (Note: The dataset is excluded from this repository due to size constraints.) Dataset reference: [data](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018)

---

## Project Highlights

1. **Optimizations**:
   - Batch processing for sentiment analysis and summarization using APIs to improve scalability and reduce costs.
   - Leveraged GPU for efficient BERT-based sentiment analysis.

2. **LLM Experimentation**:
   - Explored multiple LLM APIs (Gemini, OpenAI, GROQ) to evaluate performance and results.

3. **Healthcare-Focused Insights**:
   - Generated valuable insights into drug efficacy and user satisfaction, providing a foundation for healthcare analytics.

---

## Repository Structure

```bash
.
drug-review-analysis/
├── README.md               # Project documentation (this file)
├── data                    # Placeholder for datasets (not included in the repository)
├── sentiment_analysis.ipynb        # Sentiment analysis experiments (BERT, Gemini, Groq, text summarization)
├── BP_review_analysis.ipynb        # Text summarization experiments
├── Drug_review.ipynb               # Statistical insights (Top drugs, conditions, word-cloud, TF-IDF, Topic Modeling)
└── bigram_analysis.ipynb           # Bigram analysis
```
### Prerequisites

- **Python 3.8 or higher**
- Required Python libraries:
  - `spacy`
  - `transformers`
  - `pandas`
  - `numpy`
  - `pytorch`
  - `plotly`


This project is licensed under the MIT License.
