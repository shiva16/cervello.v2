import pandas as pd
import requests

df = pd.read_csv(r"C:\Users\ablaz\Desktop\PS\WorkingCode\summary.csv")
df.head()

summlen = len(df["Summary"])
print(summlen)

#Summarized text of the entire corpus, using Deep AI API, for quicker extractive text summarization
for i in range(1, summlen):
    #print(df["Summary"][i])
    #print("\n")
    r = requests.post(
        "https://api.deepai.org/api/summarization",
        data={
            'text': df["Summary"][i]
        },
        headers={'api-key': '19be1d69-b1bd-462f-83f9-44a53a6e699e'}
    )
    print(r.json())