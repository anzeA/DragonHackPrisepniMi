import contextlib
import io
import logging
import os
import tempfile
import time
import traceback
from PIL import Image
import feedparser
import numpy as np
import pandas as pd
import requests
import torch
import yaml
from nemo.core.classes.modelPT import ModelPT
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer

BASEDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    #encoding="utf-8",
    handlers=[
        logging.FileHandler('podcasts.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logging.info('Base directory: ' + BASEDIR)
if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
    logging.info("AMP enabled!\n")
    autocast = torch.cuda.amp.autocast
else:
    @contextlib.contextmanager
    def autocast():
        yield


def load_model(model_dir):
    with open(os.path.join(model_dir, 'model.info')) as f:
        _model_info = yaml.safe_load(f)
    print(_model_info)
    am = f"{_model_info['info']['am']['framework'].partition(':')[-1].replace(':', '_')}.{_model_info['info']['am']['framework'].partition(':')[0]}"
    _model_path = os.path.join(model_dir, am)
    model = ModelPT.restore_from(_model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.freeze()
    model.eval()
    return model


model = load_model(os.path.join(BASEDIR, 'transcribe', 'model'))
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device="cuda" if torch.cuda.is_available() else "cpu")

def get_db_metadata():
    csv_file = os.path.join(BASEDIR, 'data', 'podcasts_meta.csv')
    cols = {"title_pretty":str,'episode_title': str, 'podcast_name': str, 'link_homepage': str, 'link_mp3': str, 'description': str, 'published':str}

    if os.path.isfile(csv_file):
        return pd.read_csv(csv_file,index_col=False,dtype=cols)
    else:
        print('Create new db')
        return pd.DataFrame(columns=["title_pretty", 'episode_title', 'podcast_name', 'link_homepage','link_mp3','description', 'published'],dtype=cols)

def get_db_transcribe():
    csv_file = os.path.join(BASEDIR, 'data', 'all_transcriptions.csv')
    cols = {'title': str, 'podcast': str, 'start_time':int,'end_time':int, 'transcription': str}
    cols.update({f'emb_{i}':np.float32 for i in range(384)})

    if os.path.isfile(csv_file):
        return pd.read_csv(csv_file,index_col=False,dtype=cols)
    else:
        return pd.DataFrame(columns=list(cols.keys()),dtype=cols)


def transcribe(audio, chunk_length_s=30, batch_size=8) -> str:
    assert isinstance(audio, AudioSegment) and isinstance(chunk_length_s, int) and isinstance(batch_size, int)
    with autocast():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # file to one minute chunks also take care of last chunk

        output_sample_width = 2  # 16-bit
        output_channels = 1  # mono
        output_frame_rate = 16000  # 16 kHz
        chunk_length_ms = chunk_length_s * 1000  # chunk_length_min minutes
        audio = audio.set_frame_rate(output_frame_rate)
        audio = audio.set_channels(output_channels)
        audio = audio.set_sample_width(output_sample_width)

        chunks = []
        timestamp = []  # timestamp of each chunk in ms (start, end)
        for i in range(0, len(audio), chunk_length_ms):
            chunks.append(audio[i:i + chunk_length_ms])
            timestamp.append((i, i + chunk_length_ms))
        if len(audio) % chunk_length_ms != 0:
            start = len(audio) - (len(audio) % chunk_length_ms)
            chunks.append(audio[start:])  # no negative indexing
            timestamp.append((start, len(audio)))
        # create tmp dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            # logging.info(f'Created temporary directory {tmp_dir}')
            filenames = []
            for i, chunk in enumerate(chunks):
                file = os.path.join(tmp_dir, f'chunck_{i}.wav')
                chunk.export(file, format="wav")
                filenames.append(file)
            global model
            if model.device != device:
                model = model.to(device)
            # logging.info(f'Using device {model.device}')
            transcription = model.transcribe(filenames, batch_size=8)
        return transcription, timestamp


def get_feed_data(feed):
    feed_data = dict()
    feed_data['title'] = feed.feed.title
    feed_data['description'] = feed.feed.description
    feed_data['podcast_link'] = feed.feed.link
    feed_data['image_link'] = feed.feed.image.href
    return feed_data


def get_entry_data(entry):
    entry_data = dict()
    entry_data["title_pretty"] = entry.title
    entry_data['title'] = "".join(x for x in entry.title if x.isalnum())
    entry_data['published'] = entry.published
    entry_data['link'] = [link['href'] for link in entry.enclosures if 'audio' in link['type']]  # looking for 'audio/mpeg'
    entry_data['episode_link'] = entry.link

    if len(entry_data['link']) == 0:
        raise ValueError(f'No audio link found in {entry.enclosures}')
    else:
        entry_data['link'] = entry_data['link'][0]
    entry_data['tags'] = [tag['term'] for tag in entry.tags] if 'tags' in entry else []
    return entry_data


def download_podcast(title, published, link, tags,**kwargs):

    try:
        # Download the MP3 file
        logging.info(f'Download {title} from {link}')
        response = requests.get(link)
        audio = AudioSegment.from_file(io.BytesIO(response.content))
    except Exception as e:
        logging.error(f'Error downloading {title} from {link}: {e}')
        raise e

    transcription, timestamp = transcribe(audio, chunk_length_s=30, batch_size=8)

    emb_data = np.array(embedding_model.encode(transcription).squeeze())
    df_enb = pd.DataFrame(emb_data, columns=[f'emb_{i}' for i in range(384)])
    df = pd.DataFrame({'transcription': transcription, 'start_time': [ start for start,end in timestamp], 'end_time': [ end for start,end in timestamp]})
    df = pd.concat([df, df_enb], axis=1)
    return df

def download_image(title,image_link):
    img_path = os.path.join(BASEDIR,'data','image',"".join(x for x in title if x.isalnum()) +'.png')
    if os.path.isfile(img_path):
        logging.info('Image already exists')
        return img_path
    try:
        response = requests.get(image_link)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))
        image.save(img_path, 'PNG')

        logging.info("Image downloaded and saved as PNG successfully!")
    except requests.exceptions.RequestException as e:
        logging.info(f"Error occurred: {e}")
    except IOError as e:
        logging.info(f"Error saving the image: {e}")
    return img_path

def download_feed(feed_url):
    # Parse the RSS feed
    feed = feedparser.parse(feed_url)
    feed_basic_info = dict()

    # Extract the title, description, and image link from the feed
    try:
        feed_basic_info = get_feed_data(feed)
        #logging.info(f" feed_basic_info: {feed_basic_info}")
        download_image(feed_basic_info['title'],feed_basic_info['image_link'])

    except Exception as e:
        logging.error(f'Error parsing feed {feed_url}. {e}')
        return

    df_transcription = get_db_transcribe()
    df_metadeta = get_db_metadata()
    episode_titles = set(df_transcription['episode_title'])
    episode_titles = episode_titles.intersection(set(df_metadeta['episode_title']))

    df_metadeta = df_metadeta[df_metadeta['episode_title'].isin(episode_titles)].copy()
    df_transcription = df_transcription[df_transcription['episode_title'].isin(episode_titles)].copy()


    #metadata_all = pd.DataFrame(columns=['episode_title', 'podcast_name', 'link_homepage','link_mp3','description', 'published'])
    i = 0
    for entry in feed.entries:

        i+=1
        # Extract the title, description, and MP3 link from the feed entry
        try:
            entry_data = get_entry_data(entry)
        except Exception as e:
            logging.error(f'Error parsing entry {entry.title}')
            # log traceback
            logging.error(traceback.format_exc())
            continue

        #if "".join(x for x in entry_data['title'] if x.isalnum()) in set(df_transcription['episode_title']):
        #    logging.info(f'Podcast {entry_data["title"]} already exists in db')
        #    continue
        # Download the podcast episode
        try:
            df_episode = download_podcast(**entry_data)
        except Exception as e:
            logging.error(f'Error downloading podcast {entry.title}')
            # log traceback
            logging.error(traceback.format_exc())
            continue

        df_episode['episode_title'] = ''.join(x for x in entry_data['title'] if x.isalnum())

        metadata = {"episode_title_pretty": [], "episode_title": [], "podcast_name": [], "link_homepage": [],
                    "link_mp3": [], "description": [], "published": []}

        metadata["episode_title_pretty"].append(entry_data["title_pretty"])
        metadata["episode_title"].append(entry_data["title"])
        metadata["podcast_name"].append(feed_basic_info["title"])
        metadata["link_homepage"].append(entry_data["episode_link"])
        metadata["link_mp3"].append(entry_data["link"])
        metadata["description"].append(feed_basic_info["description"])
        metadata["published"].append(entry_data["published"])

        df_meta_new = pd.DataFrame(metadata)

        assert set(df_meta_new.columns) == set(df_metadeta.columns)
        assert set(df_transcription.columns) == set(df_episode.columns)

        df_metadeta = pd.concat([df_metadeta, df_meta_new],ignore_index=True)
        df_transcription = pd.concat([df_transcription, df_episode], ignore_index=True)

        df_metadeta.to_csv(os.path.join(BASEDIR, 'data', 'podcasts_meta.csv'),index=False)
        df_transcription.to_csv(os.path.join(BASEDIR, 'data', 'all_transcriptions.csv'),index=False)
        logging.info(f'Podcast {entry_data["title"]} downloaded successfully')

    return



def download(feed_urls):
    for feed_url in feed_urls:
        download_feed(feed_url)



if __name__ == '__main__':
    feed_urls = [
        'http://podcast.rtvslo.si/zoga_je_okrogla.xml',
        'https://anchor.fm/s/964a7d24/podcast/rss', # ogroje
        'https://anchor.fm/s/1c9278b0/podcast/rss',
                 'https://aidea.libsyn.com/rss', 
                  'https://anchor.fm/s/a809d410/podcast/rss', 'https://anchor.fm/s/1c9278b0/podcast/rss','https://anchor.fm/s/1401a0b8/podcast/rss',
                 'https://apparatus.si/oddaja/obod/feed/'
    ]
    download(feed_urls)
