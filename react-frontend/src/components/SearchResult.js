import * as React from 'react';
import { useTheme } from '@mui/material/styles';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';

/*
import SkipPreviousIcon from '@mui/icons-material/SkipPrevious';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import SkipNextIcon from '@mui/icons-material/SkipNext';
import LinearProgress from '@mui/material/LinearProgress';
//import ReactAudioPlayer from 'react-audio-player';
//import AudioPlayer from 'mui-audio-player-plus';
*/


export default function SearchResult({
  episodeName,
  podcastName,
  imageUrl,
  episodeDescription,
  mp3Url,
  progress,
  episodeLength,
}) {

  const theme = useTheme();
  return (
    <Card sx={{ display: 'flex', margin: "20px 5%" }}>
      <CardMedia component="img" sx={{ width: 151 }} image={imageUrl} alt="Podcast image" />
      <Box sx={{ display: 'flex', flexDirection: 'column' }}>
        <CardContent sx={{ flex: '1 0 auto' }}>
          <Typography component="div" variant="h5">
            {episodeName}
          </Typography>
          <Typography variant="subtitle1" color="text.secondary" component="div">
            {podcastName}
          </Typography>
          
          <Typography variant="subtitle2" color="text.secondary" component="div">
            {episodeDescription}
          </Typography>
        </CardContent>
        

        <div style={{marginLeft: "20px"}}>
          <audio
            controls
            preload="metadata"
            controlsList="nodownload noduration nofullscreen noremoteplayback novolumecontrols"
            src={mp3Url}
          ></audio>
        </div>
      </Box>
    </Card>
  );
}
