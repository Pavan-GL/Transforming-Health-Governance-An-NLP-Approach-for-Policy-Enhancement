import pandas as pd
import random

# Sample policy texts
positive_texts = [
    "improves public health by increasing funding for vaccinations.",
    "enhances patient safety standards.",
    "supports telehealth services, improving access for rural areas.",
    "promotes preventative care, aiming to reduce chronic disease rates.",
    "expands mental health services for underserved communities.",
]

negative_texts = [
    "has several gaps, leading to inadequate healthcare access.",
    "fails to address the mental health crisis effectively.",
    "does not provide enough resources for hospitals.",
    "has been criticized for not considering diverse community needs.",
    "creates barriers to access for marginalized populations.",
]

# Generate synthetic data
data = []
for i in range(1, 50001):
    if random.random() > 0.5:
        text = f"Policy {i} " + random.choice(positive_texts)
        label = 1
    else:
        text = f"Policy {i} " + random.choice(negative_texts)
        label = 0
    data.append({"id": i, "text": text, "label": label})

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('D:/Transforming Health Governance An NLP Approach for Policy Enhancement/data/health_policies.csv', index=False)
print("Dataset with 50,000 rows created at 'data/raw/health_policies.csv'")
