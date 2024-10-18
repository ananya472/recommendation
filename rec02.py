import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics import jaccard_score

# Load your data
data = pd.read_csv(r'C:\Users\anany\Desktop\content rec\a_data\a_combined_em.csv')  # Your combined data
embeddings = np.load(r'C:\Users\anany\Desktop\content rec\a_data\combined_embeddings.npy')  # Load your embeddings
user_data = pd.read_csv(r'C:\Users\anany\Desktop\content rec\a_data\user_data.csv')  # User preferences data

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_jaccard_similarity(set_a, set_b):
    """Compute Jaccard similarity between two sets."""
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0

def recommend_for_user(user_id, top_n=10):
    # Get user preferences
    user_rows = user_data[user_data['User_ID'] == user_id]
    
    if user_rows.empty:
        print(f"No data found for User ID {user_id}. Please check if the ID is correct.")
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0  # Return empty DataFrame and metrics

    user_row = user_rows.iloc[0]
    liked_products = set(user_row['Liked_Products'].split(', '))  # Liked products
    added_to_cart = set(user_row['Added_to_Cart'].split(', '))  # Products added to cart

    # Check if 'Purchased_Products' exists; if not, initialize as an empty set
    if 'Purchased_Products' in user_row:
        purchased_items = set(user_row['Purchased_Products'].split(', '))
    else:
        purchased_items = set()  # No purchased items available

    # Combine liked, added to cart, and purchased items
    all_interested_products = liked_products.union(added_to_cart).union(purchased_items)

    # Display interested products
    print(f"\nInterested Products for User {user_id}:")
    interested_product_titles = []  # List to store titles of interested products
    for product in all_interested_products:
        product_info = data[data['Product ID'] == product]
        if not product_info.empty:
            title = product_info.iloc[0]['Product Title']
            interested_product_titles.append(title)
            print(f"- {product} : {title}")  # Print product ID and title

    # Get embeddings for interested products
    liked_embeddings = []
    for product_id in all_interested_products:
        product_index = data[data['Product ID'] == product_id].index
        if not product_index.empty:
            liked_embeddings.append(embeddings[product_index[0]])
    
    if not liked_embeddings:
        print(f"No interested products found for User {user_id}.")
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0

    liked_embeddings = np.array(liked_embeddings)
    user_embedding = np.mean(liked_embeddings, axis=0).reshape(1, -1)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    faiss.normalize_L2(user_embedding)

    # Perform similarity search using FAISS
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    distances_cosine, indices_cosine = index.search(user_embedding, top_n)

    recommendations = []
    jaccard_scores = []

    for i in range(top_n):
        product_id = data.iloc[indices_cosine[0][i]]['Product ID']
        title = data.iloc[indices_cosine[0][i]]['Product Title']
        distance_cosine = distances_cosine[0][i]
        recommendations.append({'User_ID': user_id, 'Product ID': product_id, 'Cosine Distance': distance_cosine, 'Title': title})

        product_features = set(data.iloc[indices_cosine[0][i]]['Features'].split(', '))
        liked_products_normalized = {feature.strip().lower() for feature in all_interested_products}  # Normalize features
        product_features_normalized = {feature.strip().lower() for feature in product_features}  # Normalize features
        jaccard_score_value = compute_jaccard_similarity(liked_products_normalized, product_features_normalized)
        jaccard_scores.append(jaccard_score_value)

    # Combine Jaccard scores into recommendations
    for idx, score in enumerate(jaccard_scores):
        recommendations[idx]['Jaccard Score'] = score

    recommendations_df = pd.DataFrame(recommendations)

    # Sort by combined score
    recommendations_df['Combined Score'] = recommendations_df['Cosine Distance'] * 0.5 + recommendations_df['Jaccard Score'] * 0.5
    recommendations_df.sort_values(by='Combined Score', ascending=False, inplace=True)

    # Evaluation Metrics
    recommended_products = recommendations_df['Product ID'].tolist()
    
    # True Positives (TP): Items that were liked and are recommended
    true_positives = sum(1 for product in recommended_products if product in all_interested_products)
    false_positives = len(recommended_products) - true_positives
    false_negatives = len(all_interested_products) - true_positives

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return recommendations_df, true_positives, false_positives, false_negatives, precision, recall, f1_score


# Example usage
user_id = "User_6"  # Replace with the actual user ID
recommendations, tp, fp, fn, precision, recall, f1_score = recommend_for_user(user_id)

# Display recommendations and evaluation metrics
print("\nRecommendations:")
print(recommendations)
print(f"\nEvaluation Metrics for User {user_id}:")
print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

