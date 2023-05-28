import React from 'react';
import './App.css';
import Header from './components/Header';
import Main from './components/Main';

import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

function App() {
  const lightTheme = createTheme({
    palette: {
      mode: 'light',
    },
  });
  

  return (
    <ThemeProvider theme={lightTheme}>
     <CssBaseline />
        <Header />
        <Main />
    </ThemeProvider>
  );
}

export default App;
