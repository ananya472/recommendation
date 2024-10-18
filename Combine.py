
import pandas as pd

# Load product data with correct encoding to handle BOM
products_wallpaper = pd.read_csv(r'C:\Users\anany\Desktop\content rec\a_data\wallpaper_data.csv', encoding='utf-8-sig')
products_flooring = pd.read_excel(r'C:\Users\anany\Desktop\content rec\a_data\lists_in_excel.xlsx', engine='openpyxl')

# Combine product data into a single DataFrame
products_df = pd.concat([products_wallpaper, products_flooring], ignore_index=True)

# Strip whitespace from column names
products_df.columns = products_df.columns.str.strip()

# Display combined product data
print("Combined Product Data:")
print(products_df.head())
