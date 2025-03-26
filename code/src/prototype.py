import pandas as pd
import numpy as np
from transformers import pipeline, DistilBertTokenizer, DistilBertModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# 1. Load and Preprocess Synthetic Dataset
data = {
    'Customer_ID': [101, 102, 103, 104, 105],
    'Age': [25, 34, 28, 45, 30],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Non-binary'],
    'Purchase_History': ['Electronics, Gaming', 'Luxury Apparel, Cosmetics', 
                         'Books, Online Courses', 'Home Decor, Organic Products', 
                         'Travel, Outdoor Gear'],
    'Interests': ['Tech Gadgets, AI', 'Fashion, Sustainability', 
                  'Self-improvement, Finance', 'Wellness, Art', 
                  'Adventure, Photography'],
    'Engagement_Score': [85, 60, 40, 70, 80],
    'Sentiment_Score': [0.7, 0.4, 0.6, 0.3, 0.6],
    'Social_Media_Activity': ['High', 'Medium', 'Low', 'Medium', 'High']
}
df = pd.DataFrame(data)

# 2. Sentiment Analysis Refinement (BERT-based)
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
def refine_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['score'] if result['label'] == 'POSITIVE' else -result['score']

df['Refined_Sentiment'] = df['Interests'].apply(refine_sentiment)

# 3. Feature Engineering for Recommendations
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

df['Interest_Embeddings'] = df['Interests'].apply(get_embeddings)

# 4. Recommendation Engine
product_catalog = {
    'Tech Gadgets': 'Latest AI-powered smartwatch',
    'Fashion': 'Sustainable luxury dress',
    'Self-improvement': 'Online finance course',
    'Wellness': 'Organic skincare set',
    'Adventure': 'High-end camping gear'
}

def generate_recommendation(customer_row):
    interests = customer_row['Interests'].split(', ')
    sentiment = customer_row['Refined_Sentiment']
    engagement = customer_row['Engagement_Score']
    
    # Simple logic: High engagement + positive sentiment = premium product
    for interest in interests:
        if interest in product_catalog:
            if engagement > 70 and sentiment > 0.5:
                return f"Premium: {product_catalog[interest]}"
            return product_catalog[interest]
    return "Generic product suggestion"

df['Recommendation'] = df.apply(generate_recommendation, axis=1)

# 5. Customer Segmentation for Business Insights
features = df[['Age', 'Engagement_Score', 'Refined_Sentiment']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# 6. Business Insights (e.g., Churn Risk)
df['Churn_Risk'] = df.apply(lambda row: 'High' if row['Engagement_Score'] < 50 and row['Refined_Sentiment'] < 0 else 'Low', axis=1)

# 7. Output Results
print("Personalized Recommendations and Insights:")
print(df[['Customer_ID', 'Recommendation', 'Cluster', 'Churn_Risk']])

# 8. Save to GitHub-ready format
df.to_csv('hyper_personalization_output.csv', index=False)
