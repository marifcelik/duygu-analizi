import pandas as pd
import re
from typing import Tuple
import os


class DataProcessor:
    def __init__(self, data_path: str = "data"):
        self.data_path = data_path
        self.emotion_mapping = {
            'surpriz': 'surprise',
            'Surprise': 'surprise',
            'kızgın': 'anger',
            'Anger': 'anger',
            'Happy': 'joy',
            'mutlu': 'joy',
            'üzgün': 'sadness',
            'Sadness': 'sadness',
            'korku': 'fear',
            'Fear': 'fear',
            'Disgust': 'disgust'
        }
    
    def load_data(self) -> pd.DataFrame:
        frames = []
        
        train_path = os.path.join(self.data_path, 'Emotion_dataset_train.csv')
        if os.path.exists(train_path):
            train = pd.read_csv(train_path, index_col=False)
            train.rename(columns={'Sentence':'text','Label':'label'}, inplace=True)
            train.drop(['Unnamed: 0'], inplace=True, axis='columns')
            frames.append(train)
        
        test_path = os.path.join(self.data_path, 'Emotion_dataset_test.csv')
        if os.path.exists(test_path):
            test = pd.read_csv(test_path, index_col=False)
            test.rename(columns={'Sentence':'text','Label':'label'}, inplace=True)
            test.drop(['Unnamed: 0'], inplace=True, axis='columns')
            frames.append(test)
        
        tremodata_path = os.path.join(self.data_path, 'TREMODATA.xml')
        if os.path.exists(tremodata_path):
            tremodata = pd.read_xml(tremodata_path)
            tremodata = tremodata[tremodata['Condition'] == 'Consensus']
            tremodata.drop(['ID','OriginalEmotion','Condition','VoteDistribution'], 
                          axis='columns', inplace=True)
            tremodata.rename(columns={'Entry':'text', 'ValidatedEmotion':'label'}, inplace=True)
            tremodata = tremodata.reset_index(drop=True)
            frames.append(tremodata)
        
        if frames:
            df = pd.concat(frames)
            df = df.reset_index(drop=True)
        else:
            df = pd.DataFrame({
                'text': [
                    'Bugün çok mutluyum!',
                    'Üzgün bir gün geçiriyorum.',
                    'Bu durum beni çok kızdırıyor.',
                    'Korkunç bir deneyim yaşadım.',
                    'Ne kadar şaşırtıcı bir haber!'
                ],
                'label': ['joy', 'sadness', 'anger', 'fear', 'surprise']
            })
        
        df = self._normalize_emotions(df)
        
        df['text'] = df['text'].apply(self.filter_text)
        
        return df
    
    def _normalize_emotions(self, df: pd.DataFrame) -> pd.DataFrame:
        for i in df.index:
            if df.label.iloc[i] in self.emotion_mapping:
                df.at[i, 'label'] = self.emotion_mapping[df.label.iloc[i]]
        return df
    
    def filter_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
            
        final_text = ''
        for word in str(text).split():
            if word.startswith('@'):
                continue
            elif word == 'RT':
                continue
            elif word[-3:] in ['com', 'org']:
                continue
            elif word.startswith('pic') or word.startswith('http') or word.startswith('www'):
                continue
            elif word.startswith('!') or word.startswith('&') or word.startswith('-'):
                continue
            else:
                final_text += word + ' '
        return final_text.strip()
    
    def additional_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
        df = df[df['text'].str.len() > 2]
        df = df[df['text'].notna() & (df['text'] != '')]
        
        return df.reset_index(drop=True)
    
    def get_emotion_distribution(self, df: pd.DataFrame) -> dict:
        return df['label'].value_counts().to_dict()
    
    def train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df['label']
        )
        
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
