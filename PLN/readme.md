# PLN (Natural Language Processing) Directory Overview

This directory contains several Jupyter notebooks dedicated to various aspects of Natural Language Processing (NLP), showcasing different methods and applications of NLP in data analysis and machine learning.

## Notebooks Description

- **RE.ipynb**:
  - **Overview**: Extracts and analyzes various elements from a dataset of tweets using regular expressions.
  - **Key Components**: Identification of hashtags, user mentions, emojis, dates, and times.
  - **Output**: An Excel file summarizing the frequencies of each extracted element.

- **chat.ipynb**:
  - **Overview**: Demonstrates a voice-responsive chatbot that uses OpenAI's GPT models for generating conversational responses and ElevenLabs API for voice synthesis.
  - **Key Features**: Real-time text-to-speech conversion, interactive user interface.

- **covariables.ipynb**:
  - **Overview**: Advanced text processing techniques applied to CSV data, including text cleaning, normalization, and linguistic analysis using SpaCy.
  - **Output**: Enhanced CSV files with additional linguistic data.

- **distanciaTexto.ipynb**:
  - **Overview**: Performs detailed similarity analysis between text documents using various representation techniques and similarity measures.
  - **Output**: Excel file containing similarity scores and the corresponding text excerpts.

- **wikiAnalysis.ipynb**:
  - **Overview**: Utilizes data from Wikipedia to analyze and visualize the semantics and context of specific words.
  - **Techniques**: Includes text processing with SpaCy, word sense disambiguation, and vector space modeling with Word2Vec.

Each notebook is equipped with detailed instructions and code comments to assist users in understanding and leveraging the methodologies for their specific needs.

### DaVincis Directory Overview

- **clasificador.ipynb**:
  - **Overview**: Implements a binary classification system using the Multinomial Naive Bayes algorithm to distinguish between violent and non-violent text entries.
  - **Functionality**: Includes data preprocessing from CSV files, text vectorization, model training, and performance evaluation using accuracy metrics and a confusion matrix.
  - **Output**: The notebook outputs the accuracy of the model, a detailed classification report, and visualizes the model's performance through a confusion matrix.

- **texto.ipynb**:
  - **Overview**: Demonstrates text classification using various machine learning models including Support Vector Machines (SVM), Naive Bayes, and Logistic Regression.
  - **Functionality**: Covers text preprocessing, feature extraction through vectorization, balancing the dataset, and evaluating models with F1-score and classification reports.
  - **Key Features**: Focuses on different vectorization techniques, model evaluation metrics, and handling class imbalance in datasets.
  - **Output**: Performance metrics for each model tested are provided, highlighting their effectiveness for text classification.

These notebooks provide comprehensive tools for processing and analyzing text data with advanced machine learning techniques, suitable for applications such as content moderation or academic research in natural language processing.

## Usage

To use these notebooks, ensure you have Jupyter installed and run each notebook in a Jupyter environment. Required libraries and dependencies are listed at the beginning of each notebook.

## Contributing

Contributions to improve the notebooks or add new features are welcome. Please follow the standard GitHub pull request process to submit your contributions.

## License

The content in this directory is licensed under MIT License unless otherwise specified within specific notebook files.
