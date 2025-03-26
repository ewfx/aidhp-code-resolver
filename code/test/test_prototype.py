import unittest
import pandas as pd
import numpy as np
from hyper_personalization import df, refine_sentiment, generate_recommendation, scaler, kmeans

# Assuming hyper_personalization.py contains the main code and variables are accessible

class TestHyperPersonalization(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test."""
        # Use the first row of the DataFrame as a sample customer
        self.sample_customer = df.iloc[0]
        self.sample_interests = "Tech Gadgets, AI"
        self.product_catalog = {
            'Tech Gadgets': 'Latest AI-powered smartwatch',
            'Fashion': 'Sustainable luxury dress',
            'Self-improvement': 'Online finance course',
            'Wellness': 'Organic skincare set',
            'Adventure': 'High-end camping gear'
        }

    def test_data_loading(self):
        """Test if the dataset loads correctly."""
        self.assertEqual(len(df), 5, "Dataset should have 5 customers")
        self.assertIn('Customer_ID', df.columns, "Customer_ID column missing")
        self.assertIn('Interests', df.columns, "Interests column missing")

    def test_sentiment_refinement(self):
        """Test sentiment analysis refinement."""
        sentiment = refine_sentiment(self.sample_interests)
        self.assertIsInstance(sentiment, float, "Sentiment should be a float")
        self.assertTrue(-1 <= sentiment <= 1, "Sentiment score should be between -1 and 1")

    def test_recommendation_generation(self):
        """Test recommendation generation logic."""
        recommendation = generate_recommendation(self.sample_customer)
        self.assertIsInstance(recommendation, str, "Recommendation should be a string")
        self.assertIn("smartwatch", recommendation, "Recommendation should match Tech Gadgets interest")
        self.assertTrue(recommendation.startswith("Premium"), "High engagement should yield premium recommendation")

    def test_clustering(self):
        """Test customer segmentation."""
        self.assertIn('Cluster', df.columns, "Cluster column should exist")
        self.assertTrue(df['Cluster'].nunique() <= 3, "Should have at most 3 clusters")
        self.assertTrue(all(df['Cluster'].isin([0, 1, 2])), "Cluster values should be 0, 1, or 2")

    def test_churn_risk(self):
        """Test churn risk prediction."""
        self.assertIn('Churn_Risk', df.columns, "Churn_Risk column should exist")
        self.assertTrue(all(df['Churn_Risk'].isin(['High', 'Low'])), "Churn_Risk should be High or Low")
        # Test specific case: Customer 103 (low engagement, low sentiment)
        self.assertEqual(df.loc[df['Customer_ID'] == 103, 'Churn_Risk'].values[0], 'High', 
                         "Customer 103 should have High churn risk")

    def test_output_file(self):
        """Test if output CSV is generated."""
        import os
        self.assertTrue(os.path.exists('hyper_personalization_output.csv'), 
                        "Output CSV file should be created")

if __name__ == '__main__':
    unittest.main()
