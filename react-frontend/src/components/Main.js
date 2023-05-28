import React from "react";
import TextField from '@mui/material/TextField';
import { IconButton } from '@mui/material';
import Stack from '@mui/material/Stack';
import SearchIcon from '@mui/icons-material/Search';
import ScienceIcon from '@mui/icons-material/Science';
import SearchResult from './SearchResult';


export default function Main() {
  const [searchQuery, setSearchQuery] = React.useState('');
  const [data, setData] = React.useState([]);


  const handleSearchInputChange = (event) => {
    setSearchQuery(event.target.value);
  };


  React.useEffect(() => {
    const startPage = async () => {
      try {
        const response = await fetch('http://localhost:3001/api/newest', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json'
          }
        });
        const data = await response.json();
        setData(data);
      } catch (error) {
        console.error(error);
      }
    };

    startPage();
  }, []);

  const handleSearchButtonClick = () => {
    async function fetchData () {
        try {
          const response = await fetch('http://localhost:3001/api/rank_new', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({ searchQuery })
          });
          const response_data = await response.json();
          setData(response_data);
        } catch (error) {
          console.error(error);
        }
    };
    fetchData();
}


  const handleLuckyButtonClick = async () => {
    try {
      const response = await fetch('http://localhost:3001/api/lucky', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      const data = await response.json();
      setData(data);
    } catch (error) {
      console.error(error);
    }
  };


  return (
    < main>
        <Stack 
            direction="row"
            justifyContent="center"
            paddingTop="20px"
            >
            <TextField 
                id="search-bar"
                label="Search query"
                value={searchQuery}
                onChange={handleSearchInputChange}
                onKeyPress={(event)=>{
                  if (event.key === 'Enter') {
                  handleSearchButtonClick();
                }
                }}
                
            />
            <IconButton  
                children={<SearchIcon />}
                disabled={!searchQuery} 
                onClick={handleSearchButtonClick}>
                    
            </IconButton >
            <IconButton  
                children={<ScienceIcon />}
                onClick={handleLuckyButtonClick}>
                    
            </IconButton >
        </Stack>
        <Stack 
                direction="column"
                justifyContent="center"
                
                >

                {data.map((item, index) =>{
                    return (
                        <SearchResult
                            key={`${item.podcast_name} ${item.episode_title} ${index}`}
                            episodeName={item.episode_title_pretty}
                            podcastName={item.podcast_name}
                            episodeDescription= {item.description}
                            imageUrl=  {item.image_link}//{`data:image/png;base64,${item.image}`}//"https://picsum.photos/100/100"
                            mp3Url= {item.link_mp3}
                        />
                    );
                } )}

                {false && <SearchResult
                    episodeName="Episode Title 1"
                    podcastName="Podcast Name"
                    episodeDescription= "To je podcast o programiranju."
                    imageUrl="https://picsum.photos/100/100" // this is size 100x100 
                    mp3Url= "https://traffic.libsyn.com/secure/aidea/sneti.mp3?dest-id=1173629"
                    progress={0} // Example progress value
                    episodeLength="1:30:00" // Example episode length
                />}
                

            </Stack>
    </main>
  );
}

