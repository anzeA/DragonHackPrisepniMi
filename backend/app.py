from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin
import numpy as np
import pandas as pd
import logging
import os
import base64
import io
from scipy.spatial.distance import cosine
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
@lru_cache()
def get_db_metadata():
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
@lru_cache()
def get_db_transcribe():
    csv_file = os.path.join(BASEDIR, 'data', 'all_transcriptions.csv')
    cols = {'title': str, 'podcast': str, 'start_time':int,'end_time':int, 'transcription': str}
    cols.update({f'embedding_{i}':np.float32 for i in range(384)})
    #print('read only 2000 rows')
    df = pd.read_csv(csv_file,index_col=False,dtype=cols)#,nrows=2000
    df = df.dropna()
    df["embedding"] = df.apply(lambda s: np.array([s[f"emb_{i}"] for i in range(384)]), axis=1)
    df = df.drop(columns=[f"emb_{i}" for i in range(384)])

    return df


@app.route('/')
def index():
    return 'Welcome to the Podcast Search API'

df_transcribe = get_db_transcribe()
df_metadata = get_db_metadata()


logging.info('Load embedding_model')
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')

@lru_cache()
def get_tmp_image(title):
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
    return np.array(model.encode(text).squeeze())

def get_similar_news(embedding, df, n=None, embedding_col="embedding"):
    ## get similar news according to cosine similarity
    df["similarity"] = df[embedding_col].apply(lambda x: cosine_similarity(embedding, x))
    df = df.sort_values(by="similarity", ascending=False)
    return df.head(n) if isinstance(n,int) else df

def cosine_similarity(a, b):
    a=a.squeeze()
    b=b.squeeze()
    #print(a.shape, b.shape)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/api/rank', methods=['POST'])
@cross_origin()
def rank():
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
        out = df_res.to_dict(orient='records')

        logging.info(f'Done query {query}')
        return jsonify(out)
    except Exception as e:
        logging.error(f'Error processing query {query}: {e}')
        return jsonify({'result': 'Error'})

@app.route('/api/newest', methods=['GET'])
@cross_origin()
def newest():
    global df_metadata
    df = df_metadata.sort_values(by='published', ascending=False)
    df = df.head(10).copy()
    df['start_time'] = 0
    df['similarity'] = 0
    df['image'] = df.podcast_name.apply(lambda x: get_tmp_image(x))
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(port=3001, debug=True)