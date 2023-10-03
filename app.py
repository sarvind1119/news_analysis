from flask import Flask, render_template, request
import pandas as pd
import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
import datetime
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline

app = Flask(__name__)

# Load the model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Create a summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

def clean_url(searched_item, data_filter):
    x = datetime.datetime.today()
    today = str(x)[:10]
    yesterday = str(x + datetime.timedelta(days=-1))[:10]
    this_week = str(x + datetime.timedelta(days=-7))[:10]

    if data_filter == 'today':
        time_filter = 'after%3A' + yesterday
    elif data_filter == 'this_week':
        time_filter = 'after%3A' + this_week + '+before%3A' + today
    elif data_filter == 'this_year':
        time_filter = 'after%3A' + str(x.year - 1)
    elif str(data_filter).isdigit():
        temp_time = str(x + pd.Timedelta(days=-int(data_filter)))[:10]
        time_filter = 'after%3A' + temp_time + '+before%3A' + today
    else:
        time_filter = ''

    url = f'https://news.google.com/rss/search?q={searched_item}+' + time_filter + '&hl=en-US&gl=US&ceid=US%3Aen'
    return url

def get_text(x):
    start = x.find('<p>') + 3
    end = x.find('</p>')
    return x[start:end]

def get_content_from_link(link):
    try:
        response = requests.get(link, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        article_content = soup.find('article')
        if article_content:
            return article_content.get_text(separator=' ', strip=True)
        main_content = soup.find('div', {'role': 'main'}) or soup.find('section', {'role': 'main'})
        if main_content:
            return main_content.get_text(separator=' ', strip=True)
    except Exception as e:
        print(f"Error fetching content from {link}: {e}")
    return None

def update_csv_with_content(df):
    df['content'] = df['link'].apply(get_content_from_link)
    return df

def generate_summary(text):
    # Check for NaN or empty/whitespace-only strings
    if pd.isna(text) or not text.strip():
        return "No Content"

    # Tokenize the content and check its length
    tokens = tokenizer.tokenize(text)
    if len(tokens) > 1020:  # a bit less than 1024 to be safe
        tokens = tokens[:1020]
        text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))

    # Generate the summary
    return summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

def generate_summaries(df):
    df['summary'] = df['content'].apply(generate_summary)
    return df

def get_news(search_term, data_filter=None):
    url = clean_url(search_term, data_filter)
    response = requests.get(url)
    root = ET.fromstring(response.text)

    title = [i.text for i in root.findall('.//channel/item/title')]
    link = [i.text for i in root.findall('.//channel/item/link')]
    description = [i.text for i in root.findall('.//channel/item/description')]
    pubDate = [i.text for i in root.findall('.//channel/item/pubDate')]
    source = [i.text for i in root.findall('.//channel/item/source')]

    short_description = list(map(get_text, description))

    df = pd.DataFrame({'title': title, 'link': link, 'description': short_description, 'date': pubDate, 'source': source})
    df.date = pd.to_datetime(df.date, unit='ns')

    df = update_csv_with_content(df)
    df = generate_summaries(df)
    df.to_csv(f'updated_dataset_with_summaries.csv', encoding='utf-8-sig', index=False)
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_term = request.form['search_term']
        data_filter = request.form['data_filter']
        data = get_news(search_term, data_filter)
        data_list = data.to_dict(orient='records')
        return render_template('index.html', data=data_list)

    return render_template('index.html', data=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
