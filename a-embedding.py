import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load both datasets (ensure you're using correct file reading methods)
data1 = pd.read_csv(r'C:\Users\anany\Desktop\content rec\a_data\wallpaper_data.csv')  # Replace with actual path to the first data
data2 = pd.read_excel(r'C:\Users\anany\Desktop\content rec\a_data\lists_in_excel.xlsx')  # Replace with actual path to the second data

# Select relevant columns
columns = ['Product Title', 'Price', 'Product ID', 'Category', 'Subcategory', 'Type', 
           'Short Description', 'Materials', 'Product Description', 'Likes', 'Features', 'Popularity']

# Make sure both datasets have the relevant columns
data1_selected = data1[columns]
data2_selected = data2[columns]

# Combine both datasets
combined_data = pd.concat([data1_selected, data2_selected], ignore_index=True)

# Load a pre-trained embedding model (e.g., BERT-based Sentence Transformer)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to create embeddings for text columns
def get_combined_text_embedding(row):
    # Combine relevant text columns into a single string
    combined_text = f"{row['Product Title']} {row['Short Description']} {row['Product Description']} {row['Features']}"
    # Generate embedding for the combined text
    embedding = model.encode(combined_text)
    return embedding

# Apply embedding function to each row in the DataFrame
combined_data['embedding'] = combined_data.apply(get_combined_text_embedding, axis=1)

# Convert embeddings to a NumPy array for saving
embeddings = np.array(combined_data['embedding'].tolist())

# Save the combined data to a CSV file
combined_data.to_csv(r'C:\Users\anany\Desktop\content rec\a_data\a_combined_em.csv', index=False)

# Save the embeddings to a NumPy file
np.save(r'C:\Users\anany\Desktop\content rec\a_data\combined_embeddings.npy', embeddings)

print("Data and embeddings have been saved successfully!")

