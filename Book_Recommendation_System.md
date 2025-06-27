<a href="https://colab.research.google.com/github/awssensol/bookRecomment/blob/main/Copy_of_Book_Recommendation_System.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
!pip install kaggle
```

    Requirement already satisfied: kaggle in /opt/conda/lib/python3.12/site-packages (1.7.4.5)
    Requirement already satisfied: bleach in /opt/conda/lib/python3.12/site-packages (from kaggle) (6.2.0)
    Requirement already satisfied: certifi>=14.05.14 in /opt/conda/lib/python3.12/site-packages (from kaggle) (2025.4.26)
    Requirement already satisfied: charset-normalizer in /opt/conda/lib/python3.12/site-packages (from kaggle) (3.4.2)
    Requirement already satisfied: idna in /opt/conda/lib/python3.12/site-packages (from kaggle) (3.10)
    Requirement already satisfied: protobuf in /opt/conda/lib/python3.12/site-packages (from kaggle) (5.28.3)
    Requirement already satisfied: python-dateutil>=2.5.3 in /opt/conda/lib/python3.12/site-packages (from kaggle) (2.9.0.post0)
    Requirement already satisfied: python-slugify in /opt/conda/lib/python3.12/site-packages (from kaggle) (8.0.4)
    Requirement already satisfied: requests in /opt/conda/lib/python3.12/site-packages (from kaggle) (2.32.3)
    Requirement already satisfied: setuptools>=21.0.0 in /opt/conda/lib/python3.12/site-packages (from kaggle) (80.1.0)
    Requirement already satisfied: six>=1.10 in /opt/conda/lib/python3.12/site-packages (from kaggle) (1.17.0)
    Requirement already satisfied: text-unidecode in /opt/conda/lib/python3.12/site-packages (from kaggle) (1.3)
    Requirement already satisfied: tqdm in /opt/conda/lib/python3.12/site-packages (from kaggle) (4.67.1)
    Requirement already satisfied: urllib3>=1.15.1 in /opt/conda/lib/python3.12/site-packages (from kaggle) (1.26.19)
    Requirement already satisfied: webencodings in /opt/conda/lib/python3.12/site-packages (from kaggle) (0.5.1)



```python
!pip install kagglehub
```

    Requirement already satisfied: kagglehub in /opt/conda/lib/python3.12/site-packages (0.3.12)
    Requirement already satisfied: packaging in /opt/conda/lib/python3.12/site-packages (from kagglehub) (24.2)
    Requirement already satisfied: pyyaml in /opt/conda/lib/python3.12/site-packages (from kagglehub) (6.0.2)
    Requirement already satisfied: requests in /opt/conda/lib/python3.12/site-packages (from kagglehub) (2.32.3)
    Requirement already satisfied: tqdm in /opt/conda/lib/python3.12/site-packages (from kagglehub) (4.67.1)
    Requirement already satisfied: charset_normalizer<4,>=2 in /opt/conda/lib/python3.12/site-packages (from requests->kagglehub) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.12/site-packages (from requests->kagglehub) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.12/site-packages (from requests->kagglehub) (1.26.19)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.12/site-packages (from requests->kagglehub) (2025.4.26)



```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")

print("Path to dataset files:", path)
```

    Path to dataset files: /home/sagemaker-user/.cache/kagglehub/datasets/mohamedbakhet/amazon-books-reviews/versions/1



```python
!pip install pandas boto3 transformers torch keras scikit-learn nltk

import pandas as pd
import re
import json
import boto3
import nltk
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')

```

    Requirement already satisfied: pandas in /opt/conda/lib/python3.12/site-packages (2.2.3)
    Requirement already satisfied: boto3 in /opt/conda/lib/python3.12/site-packages (1.37.1)
    Requirement already satisfied: transformers in /opt/conda/lib/python3.12/site-packages (4.51.3)
    Requirement already satisfied: torch in /opt/conda/lib/python3.12/site-packages (2.6.0)
    Requirement already satisfied: keras in /opt/conda/lib/python3.12/site-packages (3.9.2)
    Requirement already satisfied: scikit-learn in /opt/conda/lib/python3.12/site-packages (1.6.1)
    Requirement already satisfied: nltk in /opt/conda/lib/python3.12/site-packages (3.9.1)
    Requirement already satisfied: numpy>=1.26.0 in /opt/conda/lib/python3.12/site-packages (from pandas) (1.26.4)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/lib/python3.12/site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.12/site-packages (from pandas) (2024.2)
    Requirement already satisfied: tzdata>=2022.7 in /opt/conda/lib/python3.12/site-packages (from pandas) (2025.2)
    Requirement already satisfied: botocore<1.38.0,>=1.37.1 in /opt/conda/lib/python3.12/site-packages (from boto3) (1.37.1)
    Requirement already satisfied: jmespath<2.0.0,>=0.7.1 in /opt/conda/lib/python3.12/site-packages (from boto3) (1.0.1)
    Requirement already satisfied: s3transfer<0.12.0,>=0.11.0 in /opt/conda/lib/python3.12/site-packages (from boto3) (0.11.3)
    Requirement already satisfied: urllib3!=2.2.0,<3,>=1.25.4 in /opt/conda/lib/python3.12/site-packages (from botocore<1.38.0,>=1.37.1->boto3) (1.26.19)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
    Requirement already satisfied: filelock in /opt/conda/lib/python3.12/site-packages (from transformers) (3.18.0)
    Requirement already satisfied: huggingface-hub<1.0,>=0.30.0 in /opt/conda/lib/python3.12/site-packages (from transformers) (0.30.2)
    Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.12/site-packages (from transformers) (24.2)
    Requirement already satisfied: pyyaml>=5.1 in /opt/conda/lib/python3.12/site-packages (from transformers) (6.0.2)
    Requirement already satisfied: regex!=2019.12.17 in /opt/conda/lib/python3.12/site-packages (from transformers) (2024.11.6)
    Requirement already satisfied: requests in /opt/conda/lib/python3.12/site-packages (from transformers) (2.32.3)
    Requirement already satisfied: tokenizers<0.22,>=0.21 in /opt/conda/lib/python3.12/site-packages (from transformers) (0.21.1)
    Requirement already satisfied: safetensors>=0.4.3 in /opt/conda/lib/python3.12/site-packages (from transformers) (0.5.3)
    Requirement already satisfied: tqdm>=4.27 in /opt/conda/lib/python3.12/site-packages (from transformers) (4.67.1)
    Requirement already satisfied: fsspec>=2023.5.0 in /opt/conda/lib/python3.12/site-packages (from huggingface-hub<1.0,>=0.30.0->transformers) (2024.10.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /opt/conda/lib/python3.12/site-packages (from huggingface-hub<1.0,>=0.30.0->transformers) (4.13.2)
    Requirement already satisfied: setuptools in /opt/conda/lib/python3.12/site-packages (from torch) (80.1.0)
    Requirement already satisfied: sympy!=1.13.2,>=1.13.1 in /opt/conda/lib/python3.12/site-packages (from torch) (1.14.0)
    Requirement already satisfied: networkx in /opt/conda/lib/python3.12/site-packages (from torch) (3.4.2)
    Requirement already satisfied: jinja2 in /opt/conda/lib/python3.12/site-packages (from torch) (3.1.6)
    Requirement already satisfied: absl-py in /opt/conda/lib/python3.12/site-packages (from keras) (2.2.0)
    Requirement already satisfied: rich in /opt/conda/lib/python3.12/site-packages (from keras) (14.0.0)
    Requirement already satisfied: namex in /opt/conda/lib/python3.12/site-packages (from keras) (0.0.9)
    Requirement already satisfied: h5py in /opt/conda/lib/python3.12/site-packages (from keras) (3.13.0)
    Requirement already satisfied: optree in /opt/conda/lib/python3.12/site-packages (from keras) (0.15.0)
    Requirement already satisfied: ml-dtypes in /opt/conda/lib/python3.12/site-packages (from keras) (0.4.0)
    Requirement already satisfied: scipy>=1.6.0 in /opt/conda/lib/python3.12/site-packages (from scikit-learn) (1.15.2)
    Requirement already satisfied: joblib>=1.2.0 in /opt/conda/lib/python3.12/site-packages (from scikit-learn) (1.5.0)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/conda/lib/python3.12/site-packages (from scikit-learn) (3.6.0)
    Requirement already satisfied: click in /opt/conda/lib/python3.12/site-packages (from nltk) (8.1.8)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/conda/lib/python3.12/site-packages (from sympy!=1.13.2,>=1.13.1->torch) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.12/site-packages (from jinja2->torch) (3.0.2)
    Requirement already satisfied: charset_normalizer<4,>=2 in /opt/conda/lib/python3.12/site-packages (from requests->transformers) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.12/site-packages (from requests->transformers) (3.10)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.12/site-packages (from requests->transformers) (2025.4.26)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /opt/conda/lib/python3.12/site-packages (from rich->keras) (3.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /opt/conda/lib/python3.12/site-packages (from rich->keras) (2.19.1)
    Requirement already satisfied: mdurl~=0.1 in /opt/conda/lib/python3.12/site-packages (from markdown-it-py>=2.2.0->rich->keras) (0.1.2)


    2025-06-26 19:11:34.429832: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    [nltk_data] Downloading package stopwords to /home/sagemaker-
    [nltk_data]     user/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True




```python
pip install swifter
```

    Requirement already satisfied: swifter in /opt/conda/lib/python3.12/site-packages (1.4.0)
    Requirement already satisfied: pandas>=1.0.0 in /opt/conda/lib/python3.12/site-packages (from swifter) (2.2.3)
    Requirement already satisfied: psutil>=5.6.6 in /opt/conda/lib/python3.12/site-packages (from swifter) (5.9.8)
    Requirement already satisfied: dask>=2.10.0 in /opt/conda/lib/python3.12/site-packages (from dask[dataframe]>=2.10.0->swifter) (2025.4.1)
    Requirement already satisfied: tqdm>=4.33.0 in /opt/conda/lib/python3.12/site-packages (from swifter) (4.67.1)
    Requirement already satisfied: click>=8.1 in /opt/conda/lib/python3.12/site-packages (from dask>=2.10.0->dask[dataframe]>=2.10.0->swifter) (8.1.8)
    Requirement already satisfied: cloudpickle>=3.0.0 in /opt/conda/lib/python3.12/site-packages (from dask>=2.10.0->dask[dataframe]>=2.10.0->swifter) (3.1.1)
    Requirement already satisfied: fsspec>=2021.09.0 in /opt/conda/lib/python3.12/site-packages (from dask>=2.10.0->dask[dataframe]>=2.10.0->swifter) (2024.10.0)
    Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.12/site-packages (from dask>=2.10.0->dask[dataframe]>=2.10.0->swifter) (24.2)
    Requirement already satisfied: partd>=1.4.0 in /opt/conda/lib/python3.12/site-packages (from dask>=2.10.0->dask[dataframe]>=2.10.0->swifter) (1.4.2)
    Requirement already satisfied: pyyaml>=5.3.1 in /opt/conda/lib/python3.12/site-packages (from dask>=2.10.0->dask[dataframe]>=2.10.0->swifter) (6.0.2)
    Requirement already satisfied: toolz>=0.10.0 in /opt/conda/lib/python3.12/site-packages (from dask>=2.10.0->dask[dataframe]>=2.10.0->swifter) (0.12.1)
    Requirement already satisfied: pyarrow>=14.0.1 in /opt/conda/lib/python3.12/site-packages (from dask[dataframe]>=2.10.0->swifter) (19.0.1)
    Requirement already satisfied: numpy>=1.26.0 in /opt/conda/lib/python3.12/site-packages (from pandas>=1.0.0->swifter) (1.26.4)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/lib/python3.12/site-packages (from pandas>=1.0.0->swifter) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.12/site-packages (from pandas>=1.0.0->swifter) (2024.2)
    Requirement already satisfied: tzdata>=2022.7 in /opt/conda/lib/python3.12/site-packages (from pandas>=1.0.0->swifter) (2025.2)
    Requirement already satisfied: locket in /opt/conda/lib/python3.12/site-packages (from partd>=1.4.0->dask>=2.10.0->dask[dataframe]>=2.10.0->swifter) (1.0.0)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas>=1.0.0->swifter) (1.17.0)
    Note: you may need to restart the kernel to use updated packages.



```python
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Any
import swifter
from functools import lru_cache

class ReviewProcessor:
    SENTIMENT_MAPPING = {
        1: 0,  # Negative
        2: 0,  # Negative
        3: 1,  # Neutral
        4: 2,  # Positive
        5: 2   # Positive
    }

    def __init__(self, reviews_path: str, books_path: str):
        self.reviews_path = Path(reviews_path)
        self.books_path = Path(books_path)
        self.reviews_df = None
        self.books_df = None
        self.merged_df = None

        # Compile regex patterns once
        self.punctuation_pattern = re.compile(r'[^a-zA-Z\s]')
        self.spaces_pattern = re.compile(r'\s+')

        # Verify files exist
        if not self.reviews_path.exists():
            raise FileNotFoundError(f"Reviews file not found: {reviews_path}")
        if not self.books_path.exists():
            raise FileNotFoundError(f"Books file not found: {books_path}")

    def inspect_files(self):
        """
        Print the column names of both CSV files
        """
        # Read just the headers
        reviews_cols = pd.read_csv(self.reviews_path, nrows=0).columns
        books_cols = pd.read_csv(self.books_path, nrows=0).columns

        print("Reviews columns:", list(reviews_cols))
        print("Books columns:", list(books_cols))
        return reviews_cols, books_cols

    @lru_cache(maxsize=10000)
    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = self.punctuation_pattern.sub('', text)
        text = self.spaces_pattern.sub(' ', text)
        return text.strip()

    def load_data(self) -> None:
        # First inspect the columns
        reviews_cols, books_cols = self.inspect_files()

        dtypes = {
            'reviewerID': 'category',
            'asin': 'category',
            'overall': 'float32'
        }

        # Read the files with the actual column names
        self.reviews_df = pd.read_csv(self.reviews_path)
        self.books_df = pd.read_csv(self.books_path)

        # Print shape information
        print(f"Reviews shape: {self.reviews_df.shape}")
        print(f"Books shape: {self.books_df.shape}")

    def process_reviews(self) -> None:
        if 'reviewText' in self.reviews_df.columns:
            review_col = 'reviewText'
        else:
            # Try to find a similar column name or use the first text column
            text_cols = self.reviews_df.select_dtypes(include=['object']).columns
            review_col = text_cols[0] if len(text_cols) > 0 else None

        if review_col is None:
            raise ValueError("Could not find review text column")

        self.reviews_df['cleaned_review'] = (
            self.reviews_df[review_col]
            .swifter.apply(self.clean_text)
        )

        if 'overall' in self.reviews_df.columns:
            self.reviews_df['label'] = self.reviews_df['overall'].map(self.SENTIMENT_MAPPING)

    def merge_data(self) -> pd.DataFrame:
        # Identify common columns for merging
        common_cols = set(self.reviews_df.columns) & set(self.books_df.columns)
        if not common_cols:
            raise ValueError("No common columns found for merging datasets")

        merge_col = list(common_cols)[0]  # Use the first common column
        print(f"Merging on column: {merge_col}")

        self.merged_df = pd.merge(
            self.reviews_df,
            self.books_df,
            on=merge_col,
            how='left'
        )

        self.optimize_memory()
        return self.merged_df

    def optimize_memory(self) -> None:
        for df in [self.reviews_df, self.books_df, self.merged_df]:
            if df is not None:
                for col in df.select_dtypes(include=['object']).columns:
                    if df[col].nunique() / len(df) < 0.5:
                        df[col] = df[col].astype('category')

    def process(self) -> pd.DataFrame:
        self.load_data()
        self.process_reviews()
        return self.merge_data()


# Define the file paths
reviews_file_path = "/home/sagemaker-user/.cache/kagglehub/datasets/mohamedbakhet/amazon-books-reviews/versions/1/Books_rating.csv"
books_details_file_path = "/home/sagemaker-user/.cache/kagglehub/datasets/mohamedbakhet/amazon-books-reviews/versions/1/books_data.csv"

# Usage
processor = ReviewProcessor(reviews_file_path, books_details_file_path)
# First inspect the files
processor.inspect_files()
# Then process
result_df = processor.process()

# Display the first few rows of the result
print("\nFirst few rows of processed data:")
display(result_df.head())
```

    Reviews columns: ['Id', 'Title', 'Price', 'User_id', 'profileName', 'review/helpfulness', 'review/score', 'review/time', 'review/summary', 'review/text']
    Books columns: ['Title', 'description', 'authors', 'image', 'previewLink', 'publisher', 'publishedDate', 'infoLink', 'categories', 'ratingsCount']
    Reviews columns: ['Id', 'Title', 'Price', 'User_id', 'profileName', 'review/helpfulness', 'review/score', 'review/time', 'review/summary', 'review/text']
    Books columns: ['Title', 'description', 'authors', 'image', 'previewLink', 'publisher', 'publishedDate', 'infoLink', 'categories', 'ratingsCount']



```python
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
X_train, X_val, y_train, y_val = train_test_split(df["cleaned_review"], df["label"], test_size=0.2)

train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
val_encodings = tokenizer(list(X_val), truncation=True, padding=True)

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}, torch.tensor(self.labels[idx])
    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, y_train.tolist())
val_dataset = ReviewDataset(val_encodings, y_val.tolist())

bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
training_args = TrainingArguments(
    output_dir="./bert_results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_total_limit=1,
    logging_dir="./bert_logs",
)

trainer = Trainer(model=bert_model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
trainer.train()
```


```python
# Import all necessary libraries
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Any
import swifter
from functools import lru_cache
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# First, define and run the ReviewProcessor
class ReviewProcessor:
    SENTIMENT_MAPPING = {
        1: 0,  # Negative
        2: 0,  # Negative
        3: 1,  # Neutral
        4: 2,  # Positive
        5: 2   # Positive
    }

    def __init__(self, reviews_path: str, books_path: str):
        self.reviews_path = Path(reviews_path)
        self.books_path = Path(books_path)
        self.reviews_df = None
        self.books_df = None
        self.merged_df = None

        # Compile regex patterns once
        self.punctuation_pattern = re.compile(r'[^a-zA-Z\s]')
        self.spaces_pattern = re.compile(r'\s+')

        # Verify files exist
        if not self.reviews_path.exists():
            raise FileNotFoundError(f"Reviews file not found: {reviews_path}")
        if not self.books_path.exists():
            raise FileNotFoundError(f"Books file not found: {books_path}")

    @lru_cache(maxsize=10000)
    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = self.punctuation_pattern.sub('', text)
        text = self.spaces_pattern.sub(' ', text)
        return text.strip()

    def load_data(self) -> None:
        self.reviews_df = pd.read_csv(self.reviews_path)
        self.books_df = pd.read_csv(self.books_path)

        print(f"Reviews shape: {self.reviews_df.shape}")
        print(f"Books shape: {self.books_df.shape}")

    def process_reviews(self) -> None:
        if 'reviewText' in self.reviews_df.columns:
            review_col = 'reviewText'
        else:
            text_cols = self.reviews_df.select_dtypes(include=['object']).columns
            review_col = text_cols[0] if len(text_cols) > 0 else None

        if review_col is None:
            raise ValueError("Could not find review text column")

        self.reviews_df['cleaned_review'] = (
            self.reviews_df[review_col]
            .apply(self.clean_text)  # Removed swifter for simplicity
        )

        if 'overall' in self.reviews_df.columns:
            self.reviews_df['label'] = self.reviews_df['overall'].map(self.SENTIMENT_MAPPING)

    def merge_data(self) -> pd.DataFrame:
        common_cols = set(self.reviews_df.columns) & set(self.books_df.columns)
        if not common_cols:
            raise ValueError("No common columns found for merging datasets")

        merge_col = list(common_cols)[0]
        print(f"Merging on column: {merge_col}")

        self.merged_df = pd.merge(
            self.reviews_df,
            self.books_df,
            on=merge_col,
            how='left'
        )

        return self.merged_df

    def process(self) -> pd.DataFrame:
        self.load_data()
        self.process_reviews()
        return self.merge_data()

# Process the data
reviews_file_path = "/kaggle/input/amazon-books-reviews/Books_rating.csv"
books_details_file_path = "/kaggle/input/amazon-books-reviews/books_data.csv"

processor = ReviewProcessor(reviews_file_path, books_details_file_path)
df = processor.process()

# Optional: Use a smaller subset of data if needed
# df = df.sample(n=10000, random_state=42)

print("\nDataFrame shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nLabel distribution:")
print(df['label'].value_counts())

# LSTM model parameters
max_words = 5000
max_len = 300

# Tokenization
tokenizer_lstm = Tokenizer(num_words=max_words)
tokenizer_lstm.fit_on_texts(df["cleaned_review"])
X_lstm = tokenizer_lstm.texts_to_sequences(df["cleaned_review"])
X_lstm = pad_sequences(X_lstm, maxlen=max_len)
y_lstm = to_categorical(df["label"])

# Create LSTM model
lstm_model = Sequential()
lstm_model.add(Embedding(max_words, 128, input_length=max_len))
lstm_model.add(LSTM(64))
lstm_model.add(Dropout(0.3))
lstm_model.add(Dense(3, activation="softmax"))

# Compile model
lstm_model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

# Print model summary
print("\nModel Summary:")
lstm_model.summary()

# Train model
history = lstm_model.fit(X_lstm, y_lstm,
                        batch_size=64,
                        epochs=3,
                        validation_split=0.2,
                        verbose=1)

# Print final metrics
print("\nTraining completed!")
print("Final training accuracy:", history.history['accuracy'][-1])
print("Final validation accuracy:", history.history['val_accuracy'][-1])
```


```python
# Import all necessary libraries
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Any
import swifter
from functools import lru_cache
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# First, define and run the ReviewProcessor
class ReviewProcessor:
    SENTIMENT_MAPPING = {
        1: 0,  # Negative
        2: 0,  # Negative
        3: 1,  # Neutral
        4: 2,  # Positive
        5: 2   # Positive
    }

    def __init__(self, reviews_path: str, books_path: str):
        self.reviews_path = Path(reviews_path)
        self.books_path = Path(books_path)
        self.reviews_df = None
        self.books_df = None
        self.merged_df = None

        # Compile regex patterns once
        self.punctuation_pattern = re.compile(r'[^a-zA-Z\s]')
        self.spaces_pattern = re.compile(r'\s+')

        # Verify files exist
        if not self.reviews_path.exists():
            raise FileNotFoundError(f"Reviews file not found: {reviews_path}")
        if not self.books_path.exists():
            raise FileNotFoundError(f"Books file not found: {books_path}")

    @lru_cache(maxsize=10000)
    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = self.punctuation_pattern.sub('', text)
        text = self.spaces_pattern.sub(' ', text)
        return text.strip()

    def load_data(self) -> None:
        self.reviews_df = pd.read_csv(self.reviews_path)
        self.books_df = pd.read_csv(self.books_path)

        print(f"Reviews shape: {self.reviews_df.shape}")
        print(f"Books shape: {self.books_df.shape}")

    def process_reviews(self) -> None:
        if 'reviewText' in self.reviews_df.columns:
            review_col = 'reviewText'
        else:
            text_cols = self.reviews_df.select_dtypes(include=['object']).columns
            review_col = text_cols[0] if len(text_cols) > 0 else None

        if review_col is None:
            raise ValueError("Could not find review text column")

        self.reviews_df['cleaned_review'] = (
            self.reviews_df[review_col]
            .apply(self.clean_text)  # Removed swifter for simplicity
        )

        # Ensure 'overall' column exists before creating 'label'
        if 'overall' in self.reviews_df.columns:
            self.reviews_df['label'] = self.reviews_df['overall'].map(self.SENTIMENT_MAPPING)
        else:
            # Handle the case where 'overall' is missing
            print("Warning: 'overall' column not found in reviews data. 'label' column will not be created.")
            # You might want to raise an error here if 'label' is strictly required later
            # raise KeyError("'overall' column required to create 'label' column.")


    def merge_data(self) -> pd.DataFrame:
        common_cols = set(self.reviews_df.columns) & set(self.books_df.columns)
        if not common_cols:
            raise ValueError("No common columns found for merging datasets")

        merge_col = list(common_cols)[0]
        print(f"Merging on column: {merge_col}")

        self.merged_df = pd.merge(
            self.reviews_df,
            self.books_df,
            on=merge_col,
            how='left'
        )

        return self.merged_df

    def process(self) -> pd.DataFrame:
        self.load_data()
        self.process_reviews()
        return self.merge_data()

# Define the file paths
reviews_file_path = "/home/sagemaker-user/.cache/kagglehub/datasets/mohamedbakhet/amazon-books-reviews/versions/1/Books_rating.csv"
books_details_file_path = "/home/sagemaker-user/.cache/kagglehub/datasets/mohamedbakhet/amazon-books-reviews/versions/1/books_data.csv"

# Usage
processor = ReviewProcessor(reviews_file_path, books_details_file_path)
df = processor.process()

# Optional: Use a smaller subset of data if needed
# df = df.sample(n=10000, random_state=42)

print("\nDataFrame shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Before trying to access df['label'], check if it exists
if 'label' in df.columns:
    print("\nLabel distribution:")
    print(df['label'].value_counts())
else:
    print("\n'label' column not found in the processed DataFrame.")


# LSTM model parameters
max_words = 5000
max_len = 300

# Check if 'label' column exists before proceeding with LSTM model
if 'label' in df.columns:
    # Tokenization
    tokenizer_lstm = Tokenizer(num_words=max_words)
    tokenizer_lstm.fit_on_texts(df["cleaned_review"])
    X_lstm = tokenizer_lstm.texts_to_sequences(df["cleaned_review"])
    X_lstm = pad_sequences(X_lstm, maxlen=max_len)
    y_lstm = to_categorical(df["label"])

    # Create LSTM model
    lstm_model = Sequential()
    lstm_model.add(Embedding(max_words, 128, input_length=max_len))
    lstm_model.add(LSTM(64))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(Dense(3, activation="softmax"))

    # Compile model
    lstm_model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

    # Print model summary
    print("\nModel Summary:")
    lstm_model.summary()

    # Train model
    history = lstm_model.fit(X_lstm, y_lstm,
                            batch_size=64,
                            epochs=3,
                            validation_split=0.2,
                            verbose=1)

    # Print final metrics
    print("\nTraining completed!")
    print("Final training accuracy:", history.history['accuracy'][-1])
    print("Final validation accuracy:", history.history['val_accuracy'][-1])
else:
    print("\nSkipping LSTM model training as 'label' column is missing.")
```


```python
# Import all necessary libraries
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Any
# Ensure swifter is installed and imported if you want to use it later,
# but the corrected code removes swifter in the ReviewProcessor.
# !pip install swifter
# import swifter
from functools import lru_cache
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
# Import train_test_split as it's used later in the Bert section
from sklearn.model_selection import train_test_split

# First, define and run the ReviewProcessor
class ReviewProcessor:
    SENTIMENT_MAPPING = {
        1: 0,  # Negative
        2: 0,  # Negative
        3: 1,  # Neutral
        4: 2,  # Positive
        5: 2   # Positive
    }

    def __init__(self, reviews_path: str, books_path: str):
        self.reviews_path = Path(reviews_path)
        self.books_path = Path(books_path)
        self.reviews_df = None
        self.books_df = None
        self.merged_df = None

        # Compile regex patterns once
        self.punctuation_pattern = re.compile(r'[^a-zA-Z\s]')
        self.spaces_pattern = re.compile(r'\s+')

        # Verify files exist
        if not self.reviews_path.exists():
            raise FileNotFoundError(f"Reviews file not found: {reviews_path}")
        if not self.books_path.exists():
            raise FileNotFoundError(f"Books file not found: {books_path}")

    # @lru_cache(maxsize=10000) # lru_cache is not effective with pandas Series apply
    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = self.punctuation_pattern.sub('', text)
        text = self.spaces_pattern.sub(' ', text)
        return text.strip()

    def load_data(self) -> None:
        print("Loading data...")
        self.reviews_df = pd.read_csv(self.reviews_path)
        self.books_df = pd.read_csv(self.books_path)

        print(f"Reviews shape: {self.reviews_df.shape}")
        print(f"Books shape: {self.books_df.shape}")
        print("Reviews columns:", list(self.reviews_df.columns))
        print("Books columns:", list(self.books_df.columns))


    def process_reviews(self) -> None:
        print("Processing reviews...")
        review_col = None
        # Explicitly check for 'reviewText' and 'review'
        possible_review_cols = ['reviewText', 'review']
        for col in possible_review_cols:
            if col in self.reviews_df.columns:
                review_col = col
                break

        if review_col is None:
            # Fallback to finding the first object/text column if known columns aren't found
            text_cols = self.reviews_df.select_dtypes(include=['object', 'string']).columns
            if len(text_cols) > 0:
                 # Prefer columns with 'review' in the name if multiple text columns exist
                 review_col = next((c for c in text_cols if 'review' in c.lower()), text_cols[0])


        if review_col is None:
             # Raise an error if no suitable review column is found
            raise ValueError(f"Could not find a suitable review text column. Looked for {possible_review_cols} or other text columns.")
        else:
            print(f"Using '{review_col}' as the review text column.")


        # Removed swifter as it was causing issues in the original notebook cell
        self.reviews_df['cleaned_review'] = self.reviews_df[review_col].apply(self.clean_text)


        # --- MODIFIED SECTION ---
        # Check for 'review/score' or 'overall' column for sentiment label
        score_col = None
        if 'review/score' in self.reviews_df.columns:
            score_col = 'review/score'
        elif 'overall' in self.reviews_df.columns:
            score_col = 'overall'

        if score_col is not None:
            print(f"Mapping '{score_col}' column to 'label'...")
            # Convert the score column to numeric, coercing errors to NaN
            self.reviews_df[score_col] = pd.to_numeric(self.reviews_df[score_col], errors='coerce')
            # Drop rows where the score became NaN after coercion
            self.reviews_df.dropna(subset=[score_col], inplace=True)
            # Map to sentiment labels
            self.reviews_df['label'] = self.reviews_df[score_col].map(self.SENTIMENT_MAPPING)
            # Drop rows where label mapping resulted in NaN (e.g., if score had values not in mapping)
            self.reviews_df.dropna(subset=['label'], inplace=True)
            # Convert label to integer type
            self.reviews_df['label'] = self.reviews_df['label'].astype(int)
        else:
            print("Warning: Neither 'review/score' nor 'overall' column found in reviews data. Cannot create 'label' column.")
            # Create an empty 'label' column to prevent later KeyErrors if code expects it
            self.reviews_df['label'] = pd.NA # Use pandas NA for nullable integer column


        # --- END MODIFIED SECTION ---


    def merge_data(self) -> pd.DataFrame:
        print("Merging data...")
        # Identify common columns for merging
        common_cols = set(self.reviews_df.columns) & set(self.books_df.columns)
        # Exclude columns that are unlikely merge keys like 'cleaned_review', 'label', 'review/score' etc.
        # Be careful excluding too many; 'Id' is used in the data and is a valid merge key here.
        # Based on column inspection, 'Id' appears to be the merge key.
        merge_cols_to_check = [col for col in common_cols if col not in ['cleaned_review', 'label', 'review/helpfulness', 'review/score', 'review/time', 'review/summary', 'review/text', 'description', 'authors', 'image', 'previewLink', 'publisher', 'publishedDate', 'infoLink', 'categories', 'ratingsCount']]


        if not merge_cols_to_check:
            # Fallback to checking common columns without strict exclusions if primary keys aren't found
            merge_cols_to_check = list(common_cols)
            if not merge_cols_to_check:
                 raise ValueError("No suitable common columns found for merging datasets. Check 'Id' or 'asin' in both files.")
            else:
                 print(f"Falling back to checking all common columns for merge key: {merge_cols_to_check}")


        # Prioritize 'Id' or 'asin' if available (based on typical dataset structures and prior run output)
        merge_col = None
        if 'Id' in merge_cols_to_check:
             merge_col = 'Id'
        elif 'asin' in merge_cols_to_check:
             merge_col = 'asin'
        else:
            # Use the first common column if preferred ones aren't found
            merge_col = list(merge_cols_to_check)[0]


        print(f"Merging on column: {merge_col}")

        # Ensure merge column is of the same type (e.g., string) before merging
        # Coerce errors just in case, though 'Id' looks like it should be fine.
        self.reviews_df[merge_col] = self.reviews_df[merge_col].astype(str)
        self.books_df[merge_col] = self.books_df[merge_col].astype(str)


        self.merged_df = pd.merge(
            self.reviews_df,
            self.books_df,
            on=merge_col,
            how='left'
        )

        self.optimize_memory()
        return self.merged_df

    def optimize_memory(self) -> None:
        print("Optimizing memory...")
        for df_name, df in [('reviews_df', self.reviews_df), ('books_df', self.books_df), ('merged_df', self.merged_df)]:
            if df is not None:
                initial_memory = df.memory_usage(deep=True).sum() / (1024**2)
                # print(f"Initial memory usage for {df_name}: {initial_memory:.2f} MB") # Keep this commented unless needed for detailed debugging
                for col in df.select_dtypes(include=['object', 'string']).columns:
                    # Avoid converting columns with very high cardinality or those intended as text like 'cleaned_review'
                    if col in ['cleaned_review', 'review/text', 'review/summary']:
                         continue

                    if df[col].nunique() / len(df) < 0.5: # Adjust threshold if necessary
                        # Check for non-numeric values before converting to category if appropriate
                        # Or handle NaNs appropriately depending on downstream use
                        if not df[col].isnull().any():
                            try:
                                # Attempt to convert to category
                                df[col] = df[col].astype('category')
                            except Exception as e:
                                # Handle potential issues during conversion
                                print(f"Could not convert non-null column '{col}' to category: {e}")
                        else:
                            # If NaNs are present, convert using nullable category type
                             try:
                                 df[col] = df[col].astype('category')
                             except Exception as e:
                                 print(f"Could not convert nullable column '{col}' to category: {e}")


                final_memory = df.memory_usage(deep=True).sum() / (1024**2)
                print(f"Final memory usage for {df_name}: {final_memory:.2f} MB")


    def process(self) -> pd.DataFrame:
        self.load_data()
        self.process_reviews()
        return self.merge_data()


# Define the file paths
reviews_file_path = "/home/sagemaker-user/.cache/kagglehub/datasets/mohamedbakhet/amazon-books-reviews/versions/1/Books_rating.csv"
books_details_file_path = "/home/sagemaker-user/.cache/kagglehub/datasets/mohamedbakhet/amazon-books-reviews/versions/1/books_data.csv"

# Usage
processor = ReviewProcessor(reviews_file_path, books_details_file_path)
# You can optionally call inspect_files here to see the columns before processing
# processor.inspect_files()
result_df = processor.process()

# Assign the result to df as the rest of the code expects a dataframe named df
df = result_df

# Display the first few rows of the result
print("\nFirst few rows of processed data:")
# Use display from IPython.display if in a notebook context, otherwise use print
try:
    from IPython.display import display
    display(df.head())
except ImportError:
    print(df.head())


print("\nDataFrame shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Check if the 'label' column exists before trying to access it
if 'label' in df.columns and not df['label'].isnull().all():
    print("\nLabel distribution:")
    # Ensure 'label' column has non-null values before value_counts if needed
    print(df['label'].value_counts())
else:
    print("Error: 'label' column was not found or contains only null values in the processed DataFrame. Check the presence and format of 'review/score' or 'overall' in the original reviews data.")

# --- Remaining code for LSTM model (assuming 'label' and 'cleaned_review' exist and are suitable) ---
# The following code will still fail if 'label' or 'cleaned_review' are not available
# or if 'label' contains NaNs after processing.
# Add checks here if you want to make this part more robust.

if 'cleaned_review' in df.columns and 'label' in df.columns and not df['label'].isnull().all() and len(df['label'].unique()) > 1:
    # LSTM model parameters
    max_words = 5000
    max_len = 300

    # Tokenization
    print("\nStarting LSTM tokenization...")
    # Drop rows where cleaned_review or label is NaN. Make a copy to avoid SettingWithCopyWarning
    df_lstm = df.dropna(subset=['cleaned_review', 'label']).copy()
    if df_lstm.empty:
        print("DataFrame is empty after dropping rows with missing 'cleaned_review' or 'label'. Skipping LSTM.")
    else:
        # Ensure label is integer type for to_categorical
        df_lstm['label'] = df_lstm['label'].astype(int)

        tokenizer_lstm = Tokenizer(num_words=max_words, oov_token="<OOV>") # Added OOV token
        tokenizer_lstm.fit_on_texts(df_lstm["cleaned_review"])
        X_lstm = tokenizer_lstm.texts_to_sequences(df_lstm["cleaned_review"])
        X_lstm = pad_sequences(X_lstm, maxlen=max_len, padding='post', truncating='post') # Added padding/truncating arguments
        # Determine number of classes dynamically
        num_classes = len(df_lstm['label'].unique())
        y_lstm = to_categorical(df_lstm["label"], num_classes=num_classes) # Specify num_classes based on unique labels


        # Create LSTM model
        print("\nBuilding LSTM model...")
        lstm_model = Sequential()
        lstm_model.add(Embedding(max_words, 128, input_length=max_len))
        lstm_model.add(LSTM(64))
        lstm_model.add(Dropout(0.3))
        lstm_model.add(Dense(num_classes, activation="softmax")) # Output layer nodes match number of unique labels

        # Compile model
        lstm_model.compile(loss="categorical_crossentropy",
                          optimizer="adam",
                          metrics=["accuracy"])

        # Print model summary
        print("\nModel Summary:")
        lstm_model.summary()

        # Train model
        print("\nTraining LSTM model...")
        # Split data before training
        X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42, stratify=df_lstm['label']) # Stratify using the label series


        history = lstm_model.fit(X_train_lstm, y_train_lstm,
                                batch_size=64,
                                epochs=3, # Reduced epochs for faster testing if needed
                                validation_data=(X_val_lstm, y_val_lstm), # Use dedicated validation data
                                verbose=1)

        # Print final metrics
        print("\nLSTM Training completed!")
        print("Final training accuracy:", history.history['accuracy'][-1])
        print("Final validation accuracy:", history.history['val_accuracy'][-1])

else:
    print("\nSkipping LSTM model training due to missing 'cleaned_review' or 'label' column, all labels are null, or insufficient unique labels (must be > 1).")
```

    2025-06-26 19:13:11.552562: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


    Loading data...



```python
!pip install streamlit pandas kagglehub plotly
```

    Requirement already satisfied: streamlit in /opt/conda/lib/python3.12/site-packages (1.46.1)
    Requirement already satisfied: pandas in /opt/conda/lib/python3.12/site-packages (2.2.3)
    Requirement already satisfied: kagglehub in /opt/conda/lib/python3.12/site-packages (0.3.12)
    Requirement already satisfied: plotly in /opt/conda/lib/python3.12/site-packages (6.0.1)
    Requirement already satisfied: altair<6,>=4.0 in /opt/conda/lib/python3.12/site-packages (from streamlit) (5.5.0)
    Requirement already satisfied: blinker<2,>=1.5.0 in /opt/conda/lib/python3.12/site-packages (from streamlit) (1.9.0)
    Requirement already satisfied: cachetools<7,>=4.0 in /opt/conda/lib/python3.12/site-packages (from streamlit) (5.5.2)
    Requirement already satisfied: click<9,>=7.0 in /opt/conda/lib/python3.12/site-packages (from streamlit) (8.1.8)
    Requirement already satisfied: numpy<3,>=1.23 in /opt/conda/lib/python3.12/site-packages (from streamlit) (1.26.4)
    Requirement already satisfied: packaging<26,>=20 in /opt/conda/lib/python3.12/site-packages (from streamlit) (24.2)
    Requirement already satisfied: pillow<12,>=7.1.0 in /opt/conda/lib/python3.12/site-packages (from streamlit) (11.2.1)
    Requirement already satisfied: protobuf<7,>=3.20 in /opt/conda/lib/python3.12/site-packages (from streamlit) (5.28.3)
    Requirement already satisfied: pyarrow>=7.0 in /opt/conda/lib/python3.12/site-packages (from streamlit) (19.0.1)
    Requirement already satisfied: requests<3,>=2.27 in /opt/conda/lib/python3.12/site-packages (from streamlit) (2.32.3)
    Requirement already satisfied: tenacity<10,>=8.1.0 in /opt/conda/lib/python3.12/site-packages (from streamlit) (9.1.2)
    Requirement already satisfied: toml<2,>=0.10.1 in /opt/conda/lib/python3.12/site-packages (from streamlit) (0.10.2)
    Requirement already satisfied: typing-extensions<5,>=4.4.0 in /opt/conda/lib/python3.12/site-packages (from streamlit) (4.13.2)
    Requirement already satisfied: watchdog<7,>=2.1.5 in /opt/conda/lib/python3.12/site-packages (from streamlit) (6.0.0)
    Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /opt/conda/lib/python3.12/site-packages (from streamlit) (3.1.44)
    Requirement already satisfied: pydeck<1,>=0.8.0b4 in /opt/conda/lib/python3.12/site-packages (from streamlit) (0.9.1)
    Requirement already satisfied: tornado!=6.5.0,<7,>=6.0.3 in /opt/conda/lib/python3.12/site-packages (from streamlit) (6.4.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/lib/python3.12/site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.12/site-packages (from pandas) (2024.2)
    Requirement already satisfied: tzdata>=2022.7 in /opt/conda/lib/python3.12/site-packages (from pandas) (2025.2)
    Requirement already satisfied: jinja2 in /opt/conda/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (3.1.6)
    Requirement already satisfied: jsonschema>=3.0 in /opt/conda/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (4.23.0)
    Requirement already satisfied: narwhals>=1.14.2 in /opt/conda/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (1.38.2)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/conda/lib/python3.12/site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)
    Requirement already satisfied: smmap<6,>=3.0.1 in /opt/conda/lib/python3.12/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)
    Requirement already satisfied: charset_normalizer<4,>=2 in /opt/conda/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (1.26.19)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (2025.4.26)
    Requirement already satisfied: pyyaml in /opt/conda/lib/python3.12/site-packages (from kagglehub) (6.0.2)
    Requirement already satisfied: tqdm in /opt/conda/lib/python3.12/site-packages (from kagglehub) (4.67.1)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.12/site-packages (from jinja2->altair<6,>=4.0->streamlit) (3.0.2)
    Requirement already satisfied: attrs>=22.2.0 in /opt/conda/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.2.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /opt/conda/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2025.4.1)
    Requirement already satisfied: referencing>=0.28.4 in /opt/conda/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.36.2)
    Requirement already satisfied: rpds-py>=0.7.1 in /opt/conda/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.24.0)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)



```python
import streamlit as st
import pandas as pd
from pathlib import Path
import kagglehub
import plotly.express as px
import plotly.graph_objects as go

class ReviewProcessor:
    def __init__(self, reviews_path, books_path):
        self.reviews_path = Path(reviews_path)
        self.books_path = Path(books_path)
        self.reviews_df = None
        self.books_df = None
        self.load_data()
    
    def load_data(self):
        try:
            if not (self.reviews_path.exists() and self.books_path.exists()):
                with st.spinner('Downloading dataset...'):
                    self.download_dataset()
            
            with st.spinner('Loading data...'):
                self.reviews_df = pd.read_csv(self.reviews_path)
                self.books_df = pd.read_csv(self.books_path)
                st.success('Data loaded successfully!')
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            raise

    def download_dataset(self):
        try:
            path = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")
            self.reviews_path = Path(path) / "Books_rating.csv"
            self.books_path = Path(path) / "books_data.csv"
        except Exception as e:
            st.error(f"Error downloading dataset: {e}")
            raise

    def get_basic_stats(self):
        stats = {
            'Total Reviews': len(self.reviews_df),
            'Total Books': len(self.books_df),
            'Average Rating': round(self.reviews_df['rating'].mean(), 2),
            'Median Rating': self.reviews_df['rating'].median()
        }
        return stats

    def create_rating_distribution(self):
        rating_counts = self.reviews_df['rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title='Distribution of Ratings',
            labels={'x': 'Rating', 'y': 'Count'}
        )
        return fig

    def get_top_rated_books(self, min_reviews=100):
        # Combine reviews with book details
        book_stats = self.reviews_df.groupby('book_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        book_stats.columns = ['book_id', 'avg_rating', 'review_count']
        
        # Filter books with minimum number of reviews
        qualified_books = book_stats[book_stats['review_count'] >= min_reviews]
        
        # Merge with book details
        top_books = qualified_books.merge(self.books_df, on='book_id')
        
        # Sort by average rating
        return top_books.sort_values('avg_rating', ascending=False).head(10)

def main():
    st.set_page_config(
        page_title="Amazon Book Reviews Analysis",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Amazon Book Reviews Analysis")
    st.markdown("---")

    # Initialize ReviewProcessor
    try:
        processor = ReviewProcessor(
            reviews_path="Books_rating.csv",
            books_path="books_data.csv"
        )

        # Create sidebar
        st.sidebar.header("Analysis Options")
        min_reviews = st.sidebar.slider(
            "Minimum number of reviews for top books",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )

        # Create main layout with columns
        col1, col2 = st.columns(2)

        # Display basic stats in the first column
        with col1:
            st.subheader("📊 Basic Statistics")
            stats = processor.get_basic_stats()
            for stat_name, stat_value in stats.items():
                st.metric(label=stat_name, value=stat_value)

        # Display rating distribution in the second column
        with col2:
            st.subheader("📈 Rating Distribution")
            rating_dist = processor.create_rating_distribution()
            st.plotly_chart(rating_dist, use_container_width=True)

        # Display top rated books
        st.subheader("🏆 Top Rated Books")
        top_books = processor.get_top_rated_books(min_reviews=min_reviews)
        
        # Create a formatted table for top books
        for idx, book in top_books.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(book['image_url'] if 'image_url' in book else "https://placeholder.com/150", 
                            width=100)
                with col2:
                    st.markdown(f"**{book['title']}**")
                    st.write(f"Author: {book['author']}")
                    st.write(f"Average Rating: ⭐ {book['avg_rating']:.2f} ({book['review_count']} reviews)")
                st.markdown("---")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()

if __name__ == "__main__":
    main()
```

    2025-06-26 19:28:14.929 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:14.930 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.100 
      [33m[1mWarning:[0m to view this Streamlit app on a browser, run it with the following
      command:
    
        streamlit run /opt/conda/lib/python3.12/site-packages/ipykernel_launcher.py [ARGUMENTS]
    2025-06-26 19:28:15.101 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.102 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.103 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.103 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.104 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.105 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.105 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.106 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.112 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.400 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.401 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.402 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.402 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.403 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.403 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.404 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.905 Thread 'Thread-6': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.906 Thread 'Thread-6': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2025-06-26 19:28:15.907 Thread 'Thread-6': missing ScriptRunContext! This warning can be ignored when running in bare mode.



```python
import pandas as pd
import plotly.express as px
from pathlib import Path
import kagglehub
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')
from tqdm.notebook import tqdm
import time

class ReviewProcessor:
    def __init__(self):
        self.reviews_df = None
        self.books_df = None
        self.data_dir = Path("data")
        self.reviews_path = self.data_dir / "Books_rating.csv"
        self.books_path = self.data_dir / "books_data.csv"
        
    def load_data(self, sample_size=None):
        try:
            print("Checking for data files...")
            
            if not (self.reviews_path.exists() and self.books_path.exists()):
                print("Downloading dataset...")
                path = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")
                self.reviews_path = Path(path) / "Books_rating.csv"
                self.books_path = Path(path) / "books_data.csv"
            
            print("Loading data...")
            if sample_size:
                self.reviews_df = pd.read_csv(self.reviews_path, nrows=sample_size)
                self.books_df = pd.read_csv(self.books_path, nrows=sample_size)
            else:
                chunks = []
                for chunk in tqdm(pd.read_csv(self.reviews_path, chunksize=100000), desc="Loading reviews"):
                    chunks.append(chunk)
                self.reviews_df = pd.concat(chunks)
                self.books_df = pd.read_csv(self.books_path)
            
            # Print column names to debug
            print("\nReviews DataFrame columns:", self.reviews_df.columns.tolist())
            print("Books DataFrame columns:", self.books_df.columns.tolist())
            
            print(f"\nData loaded successfully!")
            print(f"Reviews shape: {self.reviews_df.shape}")
            print(f"Books shape: {self.books_df.shape}")
            
        except Exception as e:
            print(f"Error: {e}")
            
    def get_basic_stats(self):
        if self.reviews_df is None or self.books_df is None:
            return
            
        print("Calculating basic statistics...")
        
        # First check if 'Rating' or 'rating' exists
        rating_column = None
        if 'Rating' in self.reviews_df.columns:
            rating_column = 'Rating'
        elif 'rating' in self.reviews_df.columns:
            rating_column = 'rating'
        else:
            print("Warning: Rating column not found. Available columns:", self.reviews_df.columns.tolist())
            return
            
        stats = {
            'Total Reviews': len(self.reviews_df),
            'Total Books': len(self.books_df),
            'Average Rating': round(self.reviews_df[rating_column].mean(), 2),
            'Median Rating': self.reviews_df[rating_column].median()
        }
        
        stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
        display(HTML("<h3>Basic Statistics</h3>"))
        display(stats_df)
        
    def plot_rating_distribution(self):
        if self.reviews_df is None:
            return
            
        # Check for rating column
        rating_column = None
        if 'Rating' in self.reviews_df.columns:
            rating_column = 'Rating'
        elif 'rating' in self.reviews_df.columns:
            rating_column = 'rating'
        else:
            print("Warning: Rating column not found")
            return
            
        print("Creating rating distribution plot...")
        rating_counts = self.reviews_df[rating_column].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title='Distribution of Ratings',
            labels={'x': 'Rating', 'y': 'Count'}
        )
        fig.show()
        
    def get_top_rated_books(self, min_reviews=100):
        if self.reviews_df is None or self.books_df is None:
            return
            
        # Check for rating column
        rating_column = None
        if 'Rating' in self.reviews_df.columns:
            rating_column = 'Rating'
        elif 'rating' in self.reviews_df.columns:
            rating_column = 'rating'
        else:
            print("Warning: Rating column not found")
            return
            
        print("Analyzing top rated books...")
        book_stats = self.reviews_df.groupby('book_id').agg({
            rating_column: ['mean', 'count']
        }).reset_index()
        
        book_stats.columns = ['book_id', 'avg_rating', 'review_count']
        qualified_books = book_stats[book_stats['review_count'] >= min_reviews]
        top_books = qualified_books.merge(self.books_df, on='book_id')
        top_books = top_books.sort_values('avg_rating', ascending=False).head(10)
        
        display(HTML("<h3>Top Rated Books</h3>"))
        display(HTML(f"<p>Showing books with at least {min_reviews} reviews</p>"))
        
        display_cols = ['title', 'author', 'avg_rating', 'review_count']
        formatted_books = top_books[display_cols].copy()
        formatted_books['avg_rating'] = formatted_books['avg_rating'].round(2)
        formatted_books.columns = ['Title', 'Author', 'Average Rating', 'Number of Reviews']
        display(formatted_books)

def analyze_amazon_books(sample_size=None):
    start_time = time.time()
    
    print("Starting analysis...")
    processor = ReviewProcessor()
    
    # Load data with optional sampling
    processor.load_data(sample_size=sample_size)
    
    # Get basic statistics
    processor.get_basic_stats()
    
    # Plot rating distribution
    processor.plot_rating_distribution()
    
    # Get top rated books
    processor.get_top_rated_books(min_reviews=100)
    
    end_time = time.time()
    print(f"\nTotal analysis time: {round(end_time - start_time, 2)} seconds")

# First, let's examine the data structure
processor = ReviewProcessor()
processor.load_data(sample_size=5)
```

    Checking for data files...
    Downloading dataset...
    Loading data...
    
    Reviews DataFrame columns: ['Id', 'Title', 'Price', 'User_id', 'profileName', 'review/helpfulness', 'review/score', 'review/time', 'review/summary', 'review/text']
    Books DataFrame columns: ['Title', 'description', 'authors', 'image', 'previewLink', 'publisher', 'publishedDate', 'infoLink', 'categories', 'ratingsCount']
    
    Data loaded successfully!
    Reviews shape: (5, 10)
    Books shape: (5, 10)



```python
# Run with larger sample after confirming column names
analyze_amazon_books(sample_size=10000)
```

    Starting analysis...
    Checking for data files...
    Downloading dataset...
    Loading data...
    
    Reviews DataFrame columns: ['Id', 'Title', 'Price', 'User_id', 'profileName', 'review/helpfulness', 'review/score', 'review/time', 'review/summary', 'review/text']
    Books DataFrame columns: ['Title', 'description', 'authors', 'image', 'previewLink', 'publisher', 'publishedDate', 'infoLink', 'categories', 'ratingsCount']
    
    Data loaded successfully!
    Reviews shape: (10000, 10)
    Books shape: (10000, 10)
    Calculating basic statistics...
    Warning: Rating column not found. Available columns: ['Id', 'Title', 'Price', 'User_id', 'profileName', 'review/helpfulness', 'review/score', 'review/time', 'review/summary', 'review/text']
    Warning: Rating column not found
    Warning: Rating column not found
    
    Total analysis time: 1.65 seconds



```python
import pandas as pd
import plotly.express as px
from pathlib import Path
import requests
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')
from tqdm.notebook import tqdm
import time

class ReviewAnalyzer:
    def __init__(self):
        self.reviews_df = None
        self.books_df = None
        # Direct links to the dataset files
        self.reviews_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"
        self.books_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
        self.setup_ui()
        
    def download_data(self):
        """Download the dataset files"""
        try:
            print("Downloading reviews data...")
            self.reviews_df = pd.read_csv(self.reviews_url, nrows=self.sample_size.value)
            
            print("Downloading books data...")
            self.books_df = pd.read_csv(self.books_url)
            
            print("Data downloaded successfully!")
            print(f"Reviews shape: {self.reviews_df.shape}")
            print(f"Books shape: {self.books_df.shape}")
            
            # Display sample of the data
            print("\nSample of reviews data:")
            display(self.reviews_df.head())
            print("\nSample of books data:")
            display(self.books_df.head())
            
            return True
        except Exception as e:
            print(f"Error downloading data: {e}")
            return False

    def setup_ui(self):
        # Create UI components
        self.load_button = widgets.Button(
            description='Load Data',
            button_style='primary',
            icon='database'
        )
        
        self.sample_size = widgets.IntText(
            value=10000,
            description='Sample Size:',
            disabled=False
        )
        
        self.min_reviews = widgets.IntSlider(
            value=100,
            min=10,
            max=1000,
            step=10,
            description='Min Reviews:',
            disabled=False
        )
        
        # Create tabs
        self.stats_output = widgets.Output()
        self.dist_output = widgets.Output()
        self.books_output = widgets.Output()
        
        self.tabs = widgets.Tab(children=[
            self.stats_output,
            self.dist_output,
            self.books_output
        ])
        
        self.tabs.set_title(0, 'Basic Stats')
        self.tabs.set_title(1, 'Rating Distribution')
        self.tabs.set_title(2, 'Top Books')
        
        # Set up callbacks
        self.load_button.on_click(self.on_load_click)
        
        # Display UI
        display(widgets.VBox([
            widgets.HBox([self.load_button, self.sample_size]),
            self.min_reviews,
            self.tabs
        ]))

    def on_load_click(self, b):
        with self.stats_output:
            clear_output()
            if self.download_data():
                self.update_all_tabs()

    def update_all_tabs(self):
        self.update_stats_tab()
        self.update_distribution_tab()
        self.update_top_books_tab()

    def update_stats_tab(self):
        with self.stats_output:
            clear_output()
            if self.reviews_df is None:
                print("Please load data first")
                return
                
            stats = {
                'Total Reviews': len(self.reviews_df),
                'Total Books': len(self.books_df),
                'Average Rating': round(self.reviews_df['rating'].mean(), 2),
                'Median Rating': self.reviews_df['rating'].median()
            }
            
            stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
            display(HTML("<h3>Basic Statistics</h3>"))
            display(stats_df)

    def update_distribution_tab(self):
        with self.dist_output:
            clear_output()
            if self.reviews_df is None:
                print("Please load data first")
                return
                
            rating_counts = self.reviews_df['rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                title='Distribution of Ratings',
                labels={'x': 'Rating', 'y': 'Count'}
            )
            fig.show()

    def update_top_books_tab(self):
        with self.books_output:
            clear_output()
            if self.reviews_df is None:
                print("Please load data first")
                return
                
            # Calculate book statistics
            book_stats = self.reviews_df.groupby('book_id').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            
            book_stats.columns = ['book_id', 'avg_rating', 'review_count']
            qualified_books = book_stats[book_stats['review_count'] >= self.min_reviews.value]
            top_books = qualified_books.merge(self.books_df, on='book_id')
            top_books = top_books.sort_values('avg_rating', ascending=False).head(10)
            
            # Format the display
            display_cols = ['title', 'authors', 'avg_rating', 'review_count']
            formatted_books = top_books[display_cols].copy()
            formatted_books['avg_rating'] = formatted_books['avg_rating'].round(2)
            formatted_books.columns = ['Title', 'Author', 'Average Rating', 'Number of Reviews']
            
            display(HTML("<h3>Top Rated Books</h3>"))
            display(HTML(f"<p>Showing books with at least {self.min_reviews.value} reviews</p>"))
            display(formatted_books)

# First make sure we have all required packages
try:
    import ipywidgets
except ImportError:
    print("Installing required packages...")
    !pip install ipywidgets

# Create and display the analyzer
print("Loading Book Review Analyzer...")
analyzer = ReviewAnalyzer()
```

    Loading Book Review Analyzer...



    VBox(children=(HBox(children=(Button(button_style='primary', description='Load Data', icon='database', style=B…



```python

```
