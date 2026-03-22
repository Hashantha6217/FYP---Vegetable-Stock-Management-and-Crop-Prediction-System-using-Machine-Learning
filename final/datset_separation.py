import pandas as pd
import os

# Load dataset
df = pd.read_csv("D:\\SEMESTER 6\\final year project\\New folder (12)\\New folder (12)\\JW New\\final\\Vegetables_fruit_prices_with_climate_130000_2020_to_2025.csv")


# Clean column names (optional)
df.columns = [col.strip() for col in df.columns]

# Drop rows with missing region or commodity info
df.dropna(subset=['Region', 'fruit_Commodity', 'vegitable_Commodity'], inplace=True)

# Get unique districts
districts = df['Region'].unique()

# Create folders if not exist
os.makedirs("Fruits", exist_ok=True)
os.makedirs("Vegetables", exist_ok=True)

# Save district-wise CSVs
for district in districts:
    df_district = df[df['Region'] == district]

    # Fruit CSV
    fruits_df = df_district[['Date', 'Region', 'Temperature', 'Rainfall (mm)', 'Humidity (%)',
                             'Crop Yield Impact Score', 'fruit_Commodity', 'fruit_Price per Unit (LKR/kg)']]
    fruits_path = f"Fruits/{district.strip().replace(' ', '_')}_fruits.csv"
    fruits_df.to_csv(fruits_path, index=False)

    # Vegetable CSV
    veg_df = df_district[['Date', 'Region', 'Temperature', 'Rainfall (mm)', 'Humidity (%)',
                          'Crop Yield Impact Score', 'vegitable_Commodity', 'vegitable_Price per Unit (LKR/kg)']]
    veg_path = f"Vegetables/{district.strip().replace(' ', '_')}_vegetables.csv"
    veg_df.to_csv(veg_path, index=False)

print("✅ All 50 files created successfully in 'Fruits/' and 'Vegetables/' folders.")flutter RuntimeError33