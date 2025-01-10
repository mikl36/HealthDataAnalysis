import pandas as pd
import numpy as np

# tyhjä DataFrame
columns = ["ikalk", "sp", "bmi", "systbp2", "diastbp2", "kol", "kol_hdl"]
n_samples = 40  # näytemäärä
test_data = pd.DataFrame(columns=columns)

# satunnaiset arvot
for i in range(n_samples):
    age_group = np.random.choice(["30-39", "40-49", "50-59", "60-69", "70-79", "80-110"])
    sex = np.random.choice([1, 2])
    bmi = round(np.random.uniform(16, 40), 2)
    systolic_bp = round(np.random.uniform(90, 160), 2)
    diastolic_bp = round(np.random.uniform(50, 100), 2)
    total_cholesterol = round(np.random.uniform(3, 7), 2)
    hdl_cholesterol = round(np.random.uniform(0.7, 2), 2)

    test_data.loc[i] = [age_group, sex, bmi, systolic_bp, diastolic_bp, total_cholesterol, hdl_cholesterol]

# tallenna
test_data.to_csv("testiaineisto.csv", index=False)
