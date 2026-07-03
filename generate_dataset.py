import numpy as np
import pandas as pd

np.random.seed(42)

crops = {
    'rice':       dict(N=(60,100), P=(30,60),  K=(30,60),  temp=(20,35), humidity=(60,90), ph=(5.5,7.0), rainfall=(150,300)),
    'maize':      dict(N=(60,100), P=(50,80),  K=(50,80),  temp=(18,35), humidity=(50,75), ph=(5.5,7.5), rainfall=(50,150)),
    'chickpea':   dict(N=(20,50),  P=(50,80),  K=(50,80),  temp=(15,30), humidity=(14,40), ph=(6.0,8.0), rainfall=(30,100)),
    'kidneybeans':dict(N=(20,50),  P=(50,80),  K=(50,80),  temp=(15,30), humidity=(18,60), ph=(5.5,7.0), rainfall=(80,200)),
    'pigeonpeas': dict(N=(20,50),  P=(50,80),  K=(50,80),  temp=(20,35), humidity=(30,70), ph=(5.5,7.0), rainfall=(60,200)),
    'mothbeans':  dict(N=(20,50),  P=(30,60),  K=(30,60),  temp=(24,38), humidity=(25,60), ph=(6.0,8.0), rainfall=(30,80)),
    'mungbean':   dict(N=(20,50),  P=(30,60),  K=(30,60),  temp=(25,38), humidity=(60,90), ph=(6.0,7.5), rainfall=(60,150)),
    'blackgram':  dict(N=(20,50),  P=(30,60),  K=(30,60),  temp=(25,38), humidity=(60,90), ph=(5.5,7.0), rainfall=(60,150)),
    'lentil':     dict(N=(20,50),  P=(50,80),  K=(50,80),  temp=(15,25), humidity=(18,50), ph=(6.0,8.0), rainfall=(30,100)),
    'pomegranate':dict(N=(20,50),  P=(10,30),  K=(30,60),  temp=(18,38), humidity=(60,90), ph=(5.5,7.5), rainfall=(50,150)),
    'banana':     dict(N=(80,120), P=(50,80),  K=(50,80),  temp=(20,35), humidity=(60,90), ph=(5.5,7.0), rainfall=(100,300)),
    'mango':      dict(N=(20,50),  P=(20,50),  K=(30,60),  temp=(24,38), humidity=(50,80), ph=(5.5,7.5), rainfall=(50,150)),
    'grapes':     dict(N=(20,50),  P=(50,80),  K=(50,80),  temp=(8,38),  humidity=(60,90), ph=(5.5,7.0), rainfall=(50,150)),
    'watermelon': dict(N=(80,120), P=(10,30),  K=(50,80),  temp=(25,38), humidity=(60,90), ph=(5.5,7.5), rainfall=(40,100)),
    'muskmelon':  dict(N=(80,120), P=(10,30),  K=(50,80),  temp=(25,38), humidity=(60,90), ph=(6.0,7.5), rainfall=(20,60)),
    'apple':      dict(N=(20,50),  P=(100,150),K=(100,150),temp=(0,24),  humidity=(60,90), ph=(5.5,7.0), rainfall=(100,200)),
    'orange':     dict(N=(20,50),  P=(10,30),  K=(10,30),  temp=(10,35), humidity=(60,90), ph=(6.0,7.5), rainfall=(60,200)),
    'papaya':     dict(N=(40,80),  P=(10,30),  K=(50,80),  temp=(25,38), humidity=(60,90), ph=(6.0,7.5), rainfall=(100,200)),
    'coconut':    dict(N=(20,50),  P=(10,30),  K=(30,60),  temp=(20,38), humidity=(60,90), ph=(5.0,8.0), rainfall=(100,300)),
    'cotton':     dict(N=(100,140),P=(30,60),  K=(10,30),  temp=(24,38), humidity=(50,80), ph=(6.0,8.0), rainfall=(60,150)),
    'jute':       dict(N=(60,100), P=(30,60),  K=(30,60),  temp=(24,38), humidity=(60,90), ph=(6.0,7.5), rainfall=(150,250)),
    'coffee':     dict(N=(80,120), P=(30,60),  K=(30,60),  temp=(15,28), humidity=(60,90), ph=(6.0,6.5), rainfall=(150,300)),
}

rows = []
n_per_crop = 100
for crop, ranges in crops.items():
    for _ in range(n_per_crop):
        rows.append({
            'N':        np.random.randint(*ranges['N']),
            'P':        np.random.randint(*ranges['P']),
            'K':        np.random.randint(*ranges['K']),
            'temperature': round(np.random.uniform(*ranges['temp']), 2),
            'humidity':    round(np.random.uniform(*ranges['humidity']), 2),
            'ph':          round(np.random.uniform(*ranges['ph']), 2),
            'rainfall':    round(np.random.uniform(*ranges['rainfall']), 2),
            'label':       crop
        })

df = pd.DataFrame(rows)
df.to_csv('Crop_recommendation.csv', index=False)
print(f"Dataset created: {df.shape[0]} rows, {df['label'].nunique()} crops")
print(df['label'].value_counts())
