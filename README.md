# DragonHack2023


Prišepni-mi is a cutting-edge product revolutionizing the podcast discovery process. Our platform harnesses the power of semantic search and vector embeddings to transcribe and analyze podcast content, ensuring that users find the most relevant and engaging material with a simple search query.

Gone are the days of endlessly scrolling through lists or relying on vague recommendations. With Prišepni-mi, you'll be connected to the content you crave in seconds. And for podcast creators, our targeted advertising system is a game-changer. Modeled after Google Ads, Prišepni-mi allows you to boost your podcast's visibility by promoting it to the top of search results when relevant to a user's query.

In short, Prišepni-mi is the ultimate win-win for podcast enthusiasts and creators alike. Join us on our journey to revolutionize podcast discovery and unleash the true potential of this ever-growing medium!

## Repository structure

/transcribe directory contains a python script for downloading, transcribing, embedding and saving podcast data and their embeddings. You can simply add RSS feed to any new podcasts that you wish to include in the database. Running the download_feed.py will save podcast embeddings to the all_transcriptions.csv file and podcast metadata to the podcasts_meta.csv file. Additionally it will download podcast images to the /image directory in /data folder.

/backend directory contains backend app written in python and using flask to setup apis. 

/frontend directory contains all frontend components necessary for the fronted of our webapp.

To run the website simply run the ./backend/app.py and open index.html in the browser.


