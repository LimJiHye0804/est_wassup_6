from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
# Initialize the Chrome driver

def crawl_analyze():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    driver.get("https://www.melon.com/chart/index.htm")

   
    titles_elements = driver.find_elements(By.CLASS_NAME, 'ellipsis.rank01')
    titles = [title.text for title in titles_elements if title.text]

    
    singer_elements = driver.find_elements(By.CLASS_NAME, 'ellipsis.rank02')
    singers = [singer.text for singer in singer_elements if singer.text]

    
    titles = titles[:50]
    singers = singers[:50]

    
    songTagList = driver.find_elements(By.CLASS_NAME,'lst50')
    numbers = [i.get_attribute('data-song-no') for i in songTagList]


    LYRIC = []

    
    for i in numbers:
        driver.get("https://www.melon.com/song/detail.htm?songId=" + i)
        lyric_elements = driver.find_elements(By.CLASS_NAME, "lyric")
        lyrics_text = ' '.join([element.text for element in lyric_elements])
        LYRIC.append(lyrics_text)

   
    driver.quit()

    LYRIC2=[]
    for i in LYRIC:
        LYRIC2.append(i.replace("\n",""))

    numbers = numbers[:50]


    df = pd.DataFrame({
        "ID": numbers,
        "제목": titles,
        "가수": singers,
        "가사": LYRIC2
    })
    df.to_csv("../멜론TOP50 가사.csv")



    #emotion analyze with mbert
    model_name_mbert = "bert-base-multilingual-cased"
    tokenizer_mbert = AutoTokenizer.from_pretrained(model_name_mbert)
    model_mbert = AutoModelForSequenceClassification.from_pretrained(model_name_mbert, num_labels=len(["긍정","부정"]))

    df['predicted_emotion_mbert'] = df['가사'].apply(lambda lyric: predict_emotion_mbert(lyric, tokenizer_mbert, model_mbert))
    #df.drop(columns="Unnamed: 0", inplace=True)

    df.to_csv("../updated_dataset.csv")
    return df


def predict_emotion_mbert(lyric, tokenizer, model):
    inputs = tokenizer(lyric, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_emotion = ["긍정","부정"][probabilities.argmax()]
    return predicted_emotion


def random_song(emotion, df):
    if emotion in [3,6]:
        filtered_df = df[df['predicted_emotion_mbert'] == '긍정']
    else:
        filtered_df = df[df['predicted_emotion_mbert'] == '부정']

    num_samples = min(3, len(filtered_df))
    
    # Randomly select songs
    if num_samples > 0:
        return filtered_df.sample(n=num_samples)
    else:
        return df.sample(n=num_samples)