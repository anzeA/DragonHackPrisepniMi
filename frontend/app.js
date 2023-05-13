const search = () => {
  console.log('Search button clicked');
  const query = document.getElementById('search-input').value;
  fetch('http://localhost:3001/api/rank', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ query })
  })
  .then(response => response.json())
  .then(data => {
    console.log('Search results received');
    const resultsContainer = document.getElementById('search-results');
    resultsContainer.innerHTML = '';
    data.forEach(result => {
      const resultDiv = createResultElement(result);
      resultsContainer.appendChild(resultDiv);
    });
  })
  .catch(error => console.error(error));
};


const createResultElement = ({ podcast,title, description, url, similarity, image,ad, mp}) => {
  const resultDiv = document.createElement('div');
  resultDiv.classList.add('result');

  const titleHeading = document.createElement('h2');
  titleHeading.textContent = title;

  const podcastHeading = document.createElement('h3');
  podcastHeading.textContent = podcast;

  if (ad === 1) {
    console.log('Adding ad-background class');
    resultDiv.classList.add('ad-background');
  }

  if (ad === "1") {
    console.log('Adding ad-background class');
    resultDiv.classList.add('ad-background');
  }

  const descriptionParagraph = document.createElement('p');
  descriptionParagraph.textContent = description;

  const urlAnchor = document.createElement('a');
  urlAnchor.textContent = url;
  urlAnchor.href = url;
  urlAnchor.target = '_blank';  // add this line


  const similarityParagraph = document.createElement('p');
  console.log(title, description, url, similarity)
  similarityParagraph.textContent = `Similarity score: ${similarity.toFixed(2)}`;

  const imageElement = document.createElement('img');
  imageElement.src = `data:image/png;base64,${image}`;

  if (ad === 1) {
    console.log('Adding ad-background class');
    resultDiv.classList.add('ad-background');
    const titleHeading = document.createElement('h2');
    titleHeading.textContent = `test`;
  }
  
  const audioElement = new Audio("https://d3ctxlq1ktw2nl.cloudfront.net/staging/2023-3-13/323628246-44100-2-66f03bc0407c6.mp3");

  
  const playPauseButton = document.createElement('button');
  playPauseButton.textContent = 'Play';
  playPauseButton.addEventListener('click', () => {
    if (audioElement.paused) {
      audioElement.play();
      playPauseButton.textContent = 'Pause';
    } else {
      audioElement.pause();
      playPauseButton.textContent = 'Play';
    }
  });

  resultDiv.appendChild(titleHeading);
  resultDiv.appendChild(podcastHeading);
  resultDiv.appendChild(descriptionParagraph);
  resultDiv.appendChild(urlAnchor);
  resultDiv.appendChild(similarityParagraph);
  resultDiv.appendChild(imageElement);
  resultDiv.appendChild(playPauseButton);  // add this line


  return resultDiv;
};




const searchForm = document.getElementById('search-form');
searchForm.addEventListener('submit', event => {
  event.preventDefault();
  search();
});

const searchInput = document.getElementById('search-input');
searchInput.addEventListener('keydown', event => {
  if (event.key === 'Enter') {
    event.preventDefault();
    search();
  }
});
