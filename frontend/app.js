const startPage = () => {
  console.log('Start Page');
  fetch('http://localhost:3001/api/newest', {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json'
    }
  })
  .then(response => response.json())
  .then(data => {
    console.log('Search results received');
    const resultsContainer = document.getElementById('search-results');
    resultsContainer.innerHTML = '';
    data.forEach(result => {
      console.log(result)
      const resultDiv = createResultElement(result);
      resultsContainer.appendChild(resultDiv);
    });
  })
  .catch(error => console.error(error));
}

window.addEventListener("load", () => {
  console.log("tukaj")
  startPage()
})

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
      console.log(result)
      const resultDiv = createResultElement(result);
      resultsContainer.appendChild(resultDiv);
    });
  })
  .catch(error => console.error(error));
};


const createResultElement = ({ podcast,episode_title_pretty, description, link_homepage, link_mp3, similarity, image,ad, mp, start_time}) => {

  const magic_time = start_time / 1000
  const isAdd = (ad == 1)

  const resultDiv = document.createElement('div');
  resultDiv.classList.add('result');
  

  const columnDiv = document.createElement('div')
  columnDiv.classList.add('row')
  columnDiv.classList.add("justify-content-center")

  const imageDiv = document.createElement('div')
  imageDiv.classList.add("col-4")

  const textDiv = document.createElement('div')
  textDiv.classList.add("col-8")

  const imageElement = document.createElement('img');
  imageElement.src = `data:image/png;base64,${image}`;

  const titleDiv = document.createElement("div")
  titleDiv.classList.add("row")
  titleDiv.classList.add("justify-content-end")
  titleDiv.classList.add("titleDiv")

  const progressBarDiv = document.createElement("div")
  
  progressBarDiv.classList.add("col-8")

  const titleHeading = document.createElement('h2');
  titleHeading.textContent = episode_title_pretty;

  const podcastHeading = document.createElement('h3');
  podcastHeading.textContent = podcast;

  console.log(isAdd)
  if (isAdd) {
    textDiv.classList.add('ad-background');
  }


  const descriptionParagraph = document.createElement('p');
  descriptionParagraph.textContent = description;

  const urlAnchor = document.createElement('a');
  urlAnchor.classList.add("col-sm-1")
  urlAnchor.classList.add("titleDiv")
  const homepageIcon = document.createElement('i')
  homepageIcon.classList.add("bi")
  homepageIcon.classList.add("bi-house-fill")
  urlAnchor.appendChild(homepageIcon)
  urlAnchor.href = link_homepage;
  urlAnchor.target = '_blank';

  const favoriteButton = document.createElement("button")
  favoriteButton.classList.add("btn")
  favoriteButton.classList.add("col-sm-1")
  favoriteButton.classList.add("titleDiv")
  
  var starIcon = document.createElement("i")
  starIcon.classList.add("bi")
  starIcon.classList.add("bi-star")
  starIcon.classList.add("favoriteBtn")
  favoriteButton.appendChild(starIcon)

  favoriteButton.addEventListener("click", () => {
    starIcon.classList.toggle("bi-star")
    starIcon.classList.toggle("bi-star-fill")
  })


  const similarityParagraph = document.createElement('p');
  similarityParagraph.textContent = `Similarity score: ${similarity.toFixed(2)}`;


  
  var audioElement = new Audio(link_mp3);

  const bottomDiv = document.createElement("div");
  bottomDiv.classList.add("bottom");
  bottomDiv.classList.add("row");
  bottomDiv.classList.add("align-items-center")
  bottomDiv.classList.add("justify-content-center")

  const buttonsDiv = document.createElement("div");
  buttonsDiv.classList.add("col-4")
  buttonsDiv.classList.add("d-flex")
  buttonsDiv.classList.add("justify-content-center")

  const playPauseButton = document.createElement('button');
  playPauseButton.classList.add("play")
  var playIcon = document.createElement("i")
  playIcon.classList.add("bi")
  playIcon.classList.add("bi-play")

  // playPauseButton.textContent = 'Play';
  playPauseButton.appendChild(playIcon)
  playPauseButton.addEventListener('click', () => {
    playIcon.classList.toggle("bi-play")
    playIcon.classList.toggle("bi-pause")
    if (audioElement.paused) {
      audioElement.play();
    } else {
      audioElement.pause();
    }
  });

  const skipForwardButton = document.createElement("button")
  skipForwardButton.classList.add("play")
  var skipForwardIcon = document.createElement("i")
  skipForwardIcon.classList.add("bi")
  skipForwardIcon.classList.add("bi-skip-forward")
  skipForwardButton.appendChild(skipForwardIcon)

  skipForwardButton.addEventListener("click", () => {
    if (audioElement.currentTime + 30.0 >= audioElement.duration){
      audioElement.currentTime = audioElement.duration
    }
    else{
      audioElement.currentTime += 30.0
    }
  })

  const skipBackwardButton = document.createElement("button")
  skipBackwardButton.classList.add("play")
  var skipBackwardIcon = document.createElement("i")
  skipBackwardIcon.classList.add("bi")
  skipBackwardIcon.classList.add("bi-skip-backward")
  skipBackwardButton.appendChild(skipBackwardIcon)

  skipBackwardButton.addEventListener("click", () => {
    if (audioElement.currentTime - 30.0 <= 0){
      audioElement.currentTime = 0
    }
    else{
      audioElement.currentTime -= 30.0
    }
    
  })

  const magicButton = document.createElement("div")
  magicButton.classList.add("play")

  const magicIcon = document.createElement("i")
  magicIcon.classList.add("bi")
  magicIcon.classList.add("bi-lightbulb")
  magicButton.appendChild(magicIcon)

  magicButton.addEventListener("click", () => {
    audioElement.currentTime = magic_time
  })

  const progressBar = document.createElement("div");
  progressBar.classList.add("progressBar");

  const listened = document.createElement("div");
  listened.classList.add("audioListened");

  const audioLeft = document.createElement("div");
  audioLeft.classList.add("audioLeft");

  const audioText = document.createElement("div");
  audioText.classList.add("audioText");
  audioText.textContent = `-.-- / -.--`

  audioElement.addEventListener("timeupdate", () => {
    const progress = (audioElement.currentTime / audioElement.duration) * 100;
    const progressLeft = 100 - progress;
    
    audioLeft.style.width = `${progressLeft}%`;
    listened.style.width = `${progress}%`;

    const hours = Math.floor(audioElement.currentTime / (60*60));
    var minutes = Math.floor(audioElement.currentTime / 60 - hours*60);
    var seconds = Math.floor(audioElement.currentTime - minutes * 60 - hours*60*60).toLocaleString('en-US', {minimumIntegerDigits: 2, useGrouping:false});;
    minutes = minutes.toLocaleString('en-US', {minimumIntegerDigits: 2, useGrouping:false});


    const hoursWhole = Math.floor(audioElement.duration / (60*60));
    var minutesWhole = Math.floor(audioElement.duration / 60 - 60 * hoursWhole);
    var secondsWhole = Math.floor(audioElement.duration - 60*minutesWhole - 60*60*hoursWhole).toLocaleString('en-US', {minimumIntegerDigits: 2, useGrouping:false});
    minutesWhole = minutesWhole.toLocaleString('en-US', {minimumIntegerDigits: 2, useGrouping:false});

    var endString = "";

    if (hoursWhole > 0){
      endString += `${hoursWhole}:`
    }
    if (minutesWhole > 0){
      endString += `${minutesWhole}:`.toLocaleString('en-US', {minimumIntegerDigits: 2, useGrouping:false})
    }
    endString += `${secondsWhole}`

    var startString = "";

    if (hours > 0){
      startString += `${hours}:`
    }
    startString += `${minutes}:`.toLocaleString('en-US', {minimumIntegerDigits: 2, useGrouping:false});
    startString += `${seconds}`;


    audioText.textContent = `${startString} / ${endString}`
    
  });

  progressBar.addEventListener("click", (event) => {
    var startX = progressBar.getBoundingClientRect().left

    var endX = progressBar.offsetWidth * 1.0;
    var x = event.clientX
    console.log(x, endX, x / endX)
    x = x - startX
    
    var progress = x / endX

    var curTime = audioElement.duration * progress
    console.log(x, endX, curTime)
    audioElement.currentTime = curTime
  })


  progressBar.appendChild(listened);
  progressBar.appendChild(audioLeft);

  buttonsDiv.appendChild(skipBackwardButton)
  buttonsDiv.appendChild(playPauseButton)
  buttonsDiv.appendChild(skipForwardButton)
  buttonsDiv.appendChild(magicButton)

  progressBarDiv.appendChild(progressBar)
  progressBarDiv.appendChild(audioText)

  bottomDiv.appendChild(buttonsDiv)
  bottomDiv.appendChild(progressBarDiv);


  imageDiv.appendChild(imageElement);

  titleDiv.appendChild(favoriteButton)
  titleDiv.appendChild(urlAnchor)
  textDiv.appendChild(titleDiv)
  textDiv.appendChild(titleHeading);
  textDiv.appendChild(podcastHeading);
  textDiv.appendChild(descriptionParagraph);
  textDiv.appendChild(similarityParagraph);

  columnDiv.appendChild(imageDiv);
  columnDiv.appendChild(textDiv);

  resultDiv.appendChild(columnDiv)
  resultDiv.appendChild(bottomDiv)



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

