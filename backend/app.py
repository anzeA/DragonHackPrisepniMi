from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin
import numpy as np
import pandas as pd
import logging
import os
import base64
import io
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from PIL import Image    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
BASEDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
def get_db_metadata():
    '''
    Load metadata from csv file
    :return: pandas dataframe with columns: episode_title_pretty,episode_title,podcast_name,link_homepage,link_mp3,description,published
    '''
    csv_file = os.path.join(BASEDIR, 'data', 'podcasts_meta.csv')
    cols = {"title_pretty":str,'episode_title': str, 'podcast_name': str, 'link_homepage': str, 'link_mp3': str, 'description': str, 'published':str}
    df = pd.read_csv(csv_file,index_col=False,dtype=cols).dropna()
    # cast published to datetime
    df['published'] = pd.to_datetime(df['published'])
    ad_episode_titles = ["002: Mravlja-mož in osa, njam njam",
                         "#10 — Razvoj umetne inteligence in načini strojnega učenja (dr. Boris Cergol)",
                         "#67 — Astrofizika, inteligenca, zavest in izzivi človeštva (Tomaž Zwitter)",
                         "#58 — Ljubezen, vera in samozdravljenje (Mojca Fatur)",
                         "#14 — AutoGPT, digitalni agenti, ChatGPT plugini in prihodnost dela — RE:moat"]

    df["ad"] = df['episode_title_pretty'].apply(lambda x: 1 if x in ad_episode_titles else 0)
    return df

def get_db_transcribe():
    ''''
    Load transcriptions from csv file
    :return: pandas dataframe with columns: title, podcast, start_time, end_time, transcription, embedding. Where start_time and end_time are in miliseconds and embedding is a 384 dimensional numpy vector
    '''
    csv_file = os.path.join(BASEDIR, 'data', 'all_transcriptions.csv')
    cols = {'title': str, 'podcast': str, 'start_time':int,'end_time':int, 'transcription': str}
    cols.update({f'embedding_{i}':np.float32 for i in range(384)})
    #print('read only 2000 rows')
    df = pd.read_csv(csv_file,index_col=False,dtype=cols)#,nrows=2000
    df = df.dropna()
    df["embedding"] = df.apply(lambda s: np.array([s[f"emb_{i}"] for i in range(384)]), axis=1)
    df = df.drop(columns=[f"emb_{i}" for i in range(384)])
    return df

def get_db_transcribe_no_vector():
    ''''
    Load transcriptions from csv file
    :return: pandas dataframe with columns: title, podcast, start_time, end_time, transcription, embedding. Where start_time and end_time are in miliseconds and embedding is a 384 dimensional numpy vector
    '''
    csv_file = os.path.join(BASEDIR, 'data', 'all_transcriptions.csv')
    cols = {'title': str, 'podcast': str, 'start_time':int,'end_time':int, 'transcription': str}
    cols.update({f'embedding_{i}':np.float32 for i in range(384)})
    #print('read only 2000 rows')
    df = pd.read_csv(csv_file,index_col=False,dtype=cols)#,nrows=2000
    df = df.dropna()
    return df


@app.route('/')
def index():
    return 'Welcome to the Podcast Search API'

# load data
logging.info('Load data. It might take a while...')
df_transcribe = get_db_transcribe()
df_transcribe_dim_cols = get_db_transcribe_no_vector()
df_metadata = get_db_metadata()
print(df_metadata.podcast_name.unique())

logging.info('Load embedding_model')
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')

@lru_cache(maxsize=1024)
def get_tmp_image(title):
    '''
    Get image from local storage
    :param title: is podcast title (str) is should be the same as in the metadata csv file
    :return: PNG image as base64 encoded string
    '''
    if title is None:
        title = 'no_image_found'
    img_path = os.path.join(BASEDIR,'data','image',"".join(x for x in title if x.isalnum()) +'.png')
    if not os.path.isfile(img_path):
        img_path = os.path.join(BASEDIR,'data','image','no_image_found.png')
    im = Image.open(img_path)
    #im.show()
    data_img = io.BytesIO()
    im.save(data_img,"png")
    encoded_img_data = base64.b64encode(data_img.getvalue())
    return encoded_img_data.decode('utf-8')


def embed_news(text, model):
    '''
    Embed text using sentence transformer model
    :param text: text to embed
    :param model: model to use
    :return: embedding as numpy array
    '''
    return np.array(model.encode(text).squeeze())

def get_similar_news(embedding, df, n=None, embedding_col="embedding"):
    '''
    Get n most similar news to the embedding
    :param embedding: query embedding
    :param df: dataframe with embeddings
    :param n: number of similar news to return if None return all
    :param embedding_col:  name of the column with embeddings
    :return: top n similar news
    '''
    df["similarity"] = df[embedding_col].apply(lambda x: cosine_similarity(embedding, x))
    df = df.sort_values(by="similarity", ascending=False)
    return df.head(n) if isinstance(n,int) else df

def cosine_similarity(a, b):
    '''
    Compute cosine similarity between two vectors
    :param a: vector a
    :param b: vector b
    :return: cosine similarity
    '''
    a=a.squeeze()
    b=b.squeeze()
    #print(a.shape, b.shape)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/api/rank', methods=['POST'])
@cross_origin()
def rank():
    '''
    Rank news based on the query
    :return: json with ranked news
    '''

    logging.info('Got request')
    query = ''
    try:
        # get the query from the client
        query = request.json.get('query', '')
        logging.info(f'Query is: {query}')
        embedding = np.array(embedding_model.encode(query).squeeze())

        df_res_with_sim = get_similar_news(embedding, df_transcribe, embedding_col="embedding")
        df_res = df_res_with_sim[['episode_title', 'similarity']].groupby('episode_title').agg( similarity=(
            'similarity', 'median'))

        df_res = df_res.merge(df_metadata, how='left', on='episode_title').sort_values(by=['ad',"similarity"], ascending=False)

        df_res["show"] = ((df_res['ad'] == 1) & (df_res['similarity'] > 0.2)) | (df_res['ad'] == 0)
        df_res = df_res[df_res["show"] == True]
        df_res.drop_duplicates(subset=['episode_title'], inplace=True)
        df_res = df_res.reset_index().sort_values(by=['ad',"similarity"], ascending=False).head(10).copy()
        df_res = df_res.dropna()

        # dodej start pa end time
        df_time = df_res_with_sim[df_res_with_sim.episode_title.isin(df_res.episode_title)  ][['start_time','episode_title','similarity']].drop_duplicates()

        df_time2 = df_time[['episode_title','similarity']].groupby('episode_title').max().reset_index()
        df_time2 = df_time2.merge(df_time, how='left', on=['episode_title','similarity'])
        #display(df_time2)
        df_res = df_res.merge(df_time2[['episode_title','start_time']], how='left', on='episode_title')
        df_res.drop(columns=['index'], inplace=True)
        #print(df_res)
        #print(df_res.ad)
        df_res['image'] = df_res.podcast_name.apply(lambda x: get_tmp_image(x))
        print(df_res.columns)
        out = df_res.to_dict(orient='records')

        logging.info(f'Done query {query}')
        return jsonify(out)
    except Exception as e:
        logging.error(f'Error processing query {query}: {e}')
        return jsonify({'result': 'Error'})

def get_newest():
    global df_metadata
    df = df_metadata.sort_values(by='published', ascending=False)
    df = df.head(10).copy()
    df['start_time'] = 0
    df['similarity'] = 0
    df['image'] = df.podcast_name.apply(lambda x: get_tmp_image(x))
    return df.to_dict(orient='records')

df_newest = get_newest()
@app.route('/api/newest', methods=['GET'])
@cross_origin()
def newest():
    global df_newest
    return jsonify(df_newest)

@app.route('/api/similar_episodes', methods=['GET'])
@cross_origin()
def similar_episodes():
    episode_title = request.json.get('episode_title', '')
    df_transcribe =  df_transcribe_dim_cols
    global df_metadata
    df_pod = df_transcribe.groupby("episode_title").agg({f"emb_{i}": "median" for i in range(384)}).reset_index()
    df_pod["embedding"] = df_pod.apply(lambda s: np.array([s[f"emb_{i}"] for i in range(384)]), axis=1)

    embedding = df_pod[df_pod.episode_title==episode_title][[f"emb_{i}" for i in range(384)]].values.squeeze()
    df_res = get_similar_news(embedding, df_pod, embedding_col="embedding")

    df_res = df_res[["episode_title", "similarity"]].merge(df_metadata, how='left', on='episode_title').sort_values(by=["similarity"], ascending=False)

    out = df_res[1:11].copy()
    out.drop(columns=['embedding'], inplace=True)
    print(out.columns)
    out['image'] = out.podcast_name.apply(lambda x: get_tmp_image(x))
    out['ad'] = 0
    return jsonify(out.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(port=3001, debug=True)