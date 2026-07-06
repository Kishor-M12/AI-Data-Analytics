import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

n = 500
categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books", "Food & Beverage"]
regions = ["North", "South", "East", "West", "Central"]
products = {
    "Electronics": ["Laptop", "Smartphone", "Tablet", "Headphones", "Smartwatch"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Dress", "Sneakers"],
    "Home & Garden": ["Sofa", "Lamp", "Plant Pot", "Curtains", "Rug"],
    "Sports": ["Running Shoes", "Yoga Mat", "Dumbbell", "Tennis Racket", "Bicycle"],
    "Books": ["Fiction Novel", "Self-Help", "Science", "Biography", "Cooking"],
    "Food & Beverage": ["Coffee Beans", "Protein Bar", "Tea Pack", "Olive Oil", "Honey"],
}

start_date = datetime(2023, 1, 1)
rows = []
for i in range(n):
    cat = random.choice(categories)
    product = random.choice(products[cat])
    date = start_date + timedelta(days=random.randint(0, 364))
    qty = random.randint(1, 50)
    price = round(random.uniform(5.0, 1500.0), 2)
    revenue = round(qty * price, 2)
    discount = round(random.uniform(0, 0.4), 2)
    customer_age = random.randint(18, 70)
    rows.append({
        "order_id": f"ORD-{1000 + i}",
        "date": date.strftime("%Y-%m-%d"),
        "category": cat,
        "product": product,
        "region": random.choice(regions),
        "quantity": qty,
        "unit_price": price,
        "discount": discount,
        "revenue": revenue,
        "customer_age": customer_age,
        "customer_gender": random.choice(["Male", "Female", "Other"]),
        "rating": round(random.uniform(1.0, 5.0), 1),
    })

df = pd.DataFrame(rows)
# Introduce some nulls for quality check testing
df.loc[df.sample(frac=0.03).index, "rating"] = None
df.loc[df.sample(frac=0.02).index, "customer_age"] = None

df.to_csv("data/sample_ecommerce.csv", index=False)
print(f"Sample dataset saved: data/sample_ecommerce.csv ({len(df)} rows)")
print(df.head())
