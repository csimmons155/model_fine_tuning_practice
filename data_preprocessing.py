



import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import re
import argparse

from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()  
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud, STOPWORDS

from transformers import BertTokenizer

# set DEBUG to 1 to print debug information
DEBUG = 0 

class MLDataHandler:

    def __init__(self, data_path, label, test_size=0.2, stratify=True):
        self.data_path = data_path
        self.label = label
        self.test_size = test_size
        self.stratify = stratify

        self.df = pd.read_csv(self.data_path)
        self.orginal_df = self.df.copy()
    
    # Split data into training and validation sets, with stratification as an option
    def split_data(self):
        stratify_col = self.df[self.label] if self.stratify else None
        train_df, val_df = train_test_split(
            self.df, 
            test_size=self.test_size, 
            stratify=stratify_col
        )
        return train_df, val_df
    
    # Get proportion of each label in the dataset
    def get_label_counts(self):
        all_labels = Counter(self.df[self.label])
        return all_labels.most_common()
    
    # Get column name and data type of each column in the dataset
    def get_column_names(self):
        return pd.DataFrame({'Column': self.df.columns, 'Data Type': self.df.dtypes.values})
    
    # Generate a word cloud for a specific class label
    def generate_wordcloud_by_class(self, target_label):

        tag = target_label
        plt.figure(figsize = (10,3))
        subset = self.df[self.df.tag == tag]
        text = subset.title.values
        cloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=500,
                          height=300
                         ).generate(" ".join(text))
        plt.imshow(cloud)
        plt.axis('off')
        plt.title("Wordcloud of " + str(tag), fontsize=23)
        plt.tight_layout()
        plt.show()
    
    
    def clean_data(self, text):
        """
        Clean the text data by removing stopwords, punctuation, and links
        """
        text = text.lower()

        # remove stopwords
        pattern = re.compile(r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*')   
        text = pattern.sub('', text) 

        text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)
        text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
        text = re.sub(" +", " ", text)  # remove multiple spaces
        text = text.strip()  # strip white space at the ends
        text = re.sub(r"http\S+", "", text)  #  remove links

        return text

    def encode_labels(self):
        """
        Encode the labels into integers
        """
        tags = self.df[self.label].unique().tolist()
        num_tags = len(tags)
        tag_to_index = {tag: i for i, tag in enumerate(tags)}
        return tag_to_index

    def tokenize_text(self):
        """
        Tokenize the text data using BERT tokenizer
        """
        tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', return_dict=False)
        encoded_inputs = tokenizer(
            self.df['text'].tolist(),
            padding="longest",
            return_tensors='tf'
        )
        return dict(ids = encoded_inputs['input_ids'],
                   masks = encoded_inputs['attention_mask'],
                   tags = self.df[self.label].tolist())
    
    
        

def main():

    global DEBUG
    parser = argparse.ArgumentParser(description="Load and preprocess data")

    # argument for data path
    parser.add_argument(
        "--data_path",
        type=str,
        default="s3://csimmons-sandbox/ai-project-1/training_data_1.csv",
        help="Path to the data file"
    )

    # argeument for label column name
    parser.add_argument(
        "--label",
        type=str,
        default="tag",
        help="Label column name"
    )

    # add optional boolean flag for DEBUG
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    args = parser.parse_args()
    DEBUG = 1 if args.debug else 0

    
  
    dataset_object = MLDataHandler(data_path=args.data_path, label=args.label)
    
    # get proportion of all labels 
    label_counts = dataset_object.get_label_counts()
    if DEBUG:
        print(label_counts)
        print(dataset_object.get_column_names())
    

    # clean data
    dataset_object.df['text'] = dataset_object.df['title'] + " " + dataset_object.df['description']
    dataset_object.df['text'] = dataset_object.df['text'].apply(dataset_object.clean_data)
    if DEBUG:
        print(f"First row of 'text' field:\n{dataset_object.df.text.values[0]}\n")

    # drop columns
    dataset_object.df = dataset_object.df.drop(columns=['id', 'title', 'description', 'created_on'])
    dataset_object.df = dataset_object.df.dropna(subset=['tag'])
    dataset_object.df = dataset_object.df[['text', 'tag']]

    # encode labels
    class_to_index = dataset_object.encode_labels()
    dataset_object.df['tag'] = dataset_object.df['tag'].map(class_to_index)
    if DEBUG:
        print(f"First row of 'tag' field (after encoding):\n{dataset_object.df.tag.values[0]}\n")

    # tokenize text
    outputs = dataset_object.tokenize_text()
    
    # print first row of outputs
    if DEBUG:
        print(f"First row's encodings:\n{outputs['ids'][0]}\n")
        print(f"First row's mask:\n{outputs['masks'][0]}\n")
        print(f"First row's tag:\n{outputs['tags'][0]}\n")

    # split data into training and validation sets
    train_df, val_df = dataset_object.split_data()
    if DEBUG: 
        print(train_df.head())
        print(dataset_object.orginal_df.head())
    
    print("Training and Validation sets created successfully.")
    



if __name__ == "__main__":
    main()

