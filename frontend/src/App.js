import React, { useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import './App.css';

function GoMainMenuButton() {
  return (
    <Link to="/mainmenu">
      <button className='homePageButton'> Main Menu</button>
    </Link>
  );
} 

function GoHomeButton() {
  return (
    <Link to="/">
      <button className='mainMenuButton'> Home Page</button>
    </Link>
  );
}

function EasyLevelButton() {
  return (
    <Link to="/easylevel">
      <button className='mainMenuLevelsButton'>Easy Level</button>
    </Link>
  );
}

function MediumLevelButton() {
  return (
    <Link to="/mediumlevel">
      <button className='mainMenuLevelsButton'>Medium Level</button>
    </Link>
  );
}

function HardLevelButton() {
  return (
    <Link to="/hardlevel">
      <button className='mainMenuLevelsButton'>Hard Level</button>
    </Link>
  );
}

function HomePage() {
  return (
    <div className="App">
      <header className="App-header-hp">
        <p>
          Welcome to Hue by Bhavya! Let's get started.
        </p>
        <GoMainMenuButton />

      </header>
    </div>
  );
}

function MainMenu() {
  return (
    <div className="App">
      <header className="App-header">
        {/* <h1>Welcome to a recreation of the I Heart Hue Game!</h1> */}
        <div className='MM'>
          Choose a level of difficulty!
          <div className="mainMenuButtonsDiv">
            <EasyLevelButton />
            <MediumLevelButton />
            <HardLevelButton />
          </div>
          <div className = "goHomeButtonDiv">
            <GoHomeButton />
          </div>
        </div>
      </header>
    </div>
  );
}

function EasyLevel() {
  const [gridData, setGridData] = useState(null);
  const [colors, setColors] = useState([]);
  const [originalColors, setOriginalColors] = useState([]);
  const [selectedCells, setSelectedCells] = useState([]);
  const [isWin, setIsWin] = useState(false);
  const [isGridFetched, setIsGridFetched] = useState(false);
  const [showHint, setShowHint] = useState(false); //new
  
  const handleCellClick = (index) => {
    if (gridData.flat()[index] === 0) return;

    setSelectedCells(prev => {
      if (prev.includes(index)) {
        return prev.filter(i => i !== index);
      } else if (prev.length < 2) {
        return [...prev, index];
      }
      return prev;
    });
  };

  const swapCells = () => {
    if (selectedCells.length === 2) {
      const newColors = [...colors];
      [newColors[selectedCells[0]], newColors[selectedCells[1]]] = [newColors[selectedCells[1]], newColors[selectedCells[0]]];
      setColors(newColors);
      setSelectedCells([]);
      checkWinCondition(newColors);
    }
  };

  const checkWinCondition = (currentColors) => {
    const isWin = currentColors.every((color, index) => color === originalColors[index]);
    setIsWin(isWin);
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        // const response = await axios.get('/api/generate_easy_grid');
        const response = await axios.get(
          `${process.env.REACT_APP_API_BASE_URL}/api/generate_easy_grid`
        );
        setGridData(response.data.grid);
        setIsGridFetched(true);
      } catch (error) {
        console.error('Error fetching grid data:', error);
      }
    };

    if (!isGridFetched) {
      fetchData();
    }
  }, [isGridFetched]);

  const showHintPopup = () => { //new
    setShowHint(true);
    setTimeout(() => setShowHint(false), 2000);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Welcome to the easy level page!</h1>
          <div>
          {gridData ? (
            <Grid 
              gridData={gridData} 
              colors={colors} 
              setColors={setColors}
              setOriginalColors={setOriginalColors}
              selectedCells={selectedCells}
              onCellClick={handleCellClick}
            />
          ) : (
            <p>Loading grid...</p>
          )}
          </div>
          {selectedCells.length === 2 && (
            <button onClick={swapCells} className='SwapCellsButton'>Swap Cells</button>
          )}
          {isWin && <h2>Congratulations! You've won!</h2>}

          {/* Hint Button */}
          <button onClick={showHintPopup} className="hint-button">Hint</button>

          {/* Hint Popup */}
          {showHint && (
            <div className="hint-popup">
              <h3>Solution:</h3>
              <Grid 
                gridData={gridData} 
                colors={originalColors}  //Show correct solution
                setColors={() => {}}  
                selectedCells={[]} 
                onCellClick={() => {}}
              />
            </div>
          )}

        <GoMainMenuButton />
      </header>
    </div>
  );
}

function MediumLevel() {
  const [gridData, setGridData] = useState(null);
  const [colors, setColors] = useState([]);
  const [originalColors, setOriginalColors] = useState([]);
  const [selectedCells, setSelectedCells] = useState([]);
  const [isWin, setIsWin] = useState(false);
  const [isGridFetched, setIsGridFetched] = useState(false);
  const [showHint, setShowHint] = useState(false); //new
  
  const handleCellClick = (index) => {
    if (gridData.flat()[index] === 0) return;

    setSelectedCells(prev => {
      if (prev.includes(index)) {
        return prev.filter(i => i !== index);
      } else if (prev.length < 2) {
        return [...prev, index];
      }
      return prev;
    });
  };

  const swapCells = () => {
    if (selectedCells.length === 2) {
      const newColors = [...colors];
      [newColors[selectedCells[0]], newColors[selectedCells[1]]] = [newColors[selectedCells[1]], newColors[selectedCells[0]]];
      setColors(newColors);
      setSelectedCells([]);
      checkWinCondition(newColors);
    }
  };

  const checkWinCondition = (currentColors) => {
    const isWin = currentColors.every((color, index) => color === originalColors[index]);
    setIsWin(isWin);
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(
          `${process.env.REACT_APP_API_BASE_URL}/api/generate_medium_grid`
        );
        setGridData(response.data.grid);
        setIsGridFetched(true);
      } catch (error) {
        console.error('Error fetching grid data:', error);
      }
    };

    if (!isGridFetched) {
      fetchData();
    }
  }, [isGridFetched]);

  const showHintPopup = () => { //new
    setShowHint(true);
    setTimeout(() => setShowHint(false), 2000);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Welcome to the medium level page!</h1>
          <div>
          {gridData ? (
            <Grid 
              gridData={gridData} 
              colors={colors} 
              setColors={setColors}
              setOriginalColors={setOriginalColors}
              selectedCells={selectedCells}
              onCellClick={handleCellClick}
            />
          ) : (
            <p>Loading grid...</p>
          )}
          </div>
          {selectedCells.length === 2 && (
            <button onClick={swapCells} className='SwapCellsButton'>Swap Cells</button>
          )}
          {isWin && <h2>Congratulations! You've won!</h2>}

          {/* Hint Button */}
          <button onClick={showHintPopup} className="hint-button">Hint</button>

          {/* Hint Popup */}
          {showHint && (
            <div className="hint-popup">
              <h3>Solution:</h3>
              <Grid 
                gridData={gridData} 
                colors={originalColors}  //Show correct solution
                setColors={() => {}}  
                selectedCells={[]} 
                onCellClick={() => {}}
              />
            </div>
          )}

        <GoMainMenuButton />
      </header>
    </div>
  );
}

function HardLevel() {
  const [gridData, setGridData] = useState(null);
  const [colors, setColors] = useState([]);
  const [originalColors, setOriginalColors] = useState([]);
  const [selectedCells, setSelectedCells] = useState([]);
  const [isWin, setIsWin] = useState(false);
  const [isGridFetched, setIsGridFetched] = useState(false);
  const [showHint, setShowHint] = useState(false); //new
  
  const handleCellClick = (index) => {
    if (gridData.flat()[index] === 0) return;

    setSelectedCells(prev => {
      if (prev.includes(index)) {
        return prev.filter(i => i !== index);
      } else if (prev.length < 2) {
        return [...prev, index];
      }
      return prev;
    });
  };

  const swapCells = () => {
    if (selectedCells.length === 2) {
      const newColors = [...colors];
      [newColors[selectedCells[0]], newColors[selectedCells[1]]] = [newColors[selectedCells[1]], newColors[selectedCells[0]]];
      setColors(newColors);
      setSelectedCells([]);
      checkWinCondition(newColors);
    }
  };

  const checkWinCondition = (currentColors) => {
    const isWin = currentColors.every((color, index) => color === originalColors[index]);
    setIsWin(isWin);
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(
          `${process.env.REACT_APP_API_BASE_URL}/api/generate_hard_grid`
        );
        setGridData(response.data.grid);
        setIsGridFetched(true);
      } catch (error) {
        console.error('Error fetching grid data:', error);
      }
    };

    if (!isGridFetched) {
      fetchData();
    }
  }, [isGridFetched]);

  const showHintPopup = () => { //new
    setShowHint(true);
    setTimeout(() => setShowHint(false), 2000);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Welcome to the hard level page!</h1>
          <div>
          {gridData ? (
            <Grid 
              gridData={gridData} 
              colors={colors} 
              setColors={setColors}
              setOriginalColors={setOriginalColors}
              selectedCells={selectedCells}
              onCellClick={handleCellClick}
            />
          ) : (
            <p>Loading grid...</p>
          )}
          </div>
          {selectedCells.length === 2 && (
            <button onClick={swapCells} className='SwapCellsButton'>Swap Cells</button>
          )}
          {isWin && <h2>Congratulations! You've won!</h2>}

          {/* Hint Button */}
          <button onClick={showHintPopup} className="hint-button">Hint</button>

          {/* Hint Popup */}
          {showHint && (
            <div className="hint-popup">
              <h3>Solution:</h3>
              <Grid 
                gridData={gridData} 
                colors={originalColors}  //Show correct solution
                setColors={() => {}}  
                selectedCells={[]} 
                onCellClick={() => {}}
              />
            </div>
          )}

        <GoMainMenuButton />
      </header>
    </div>
  );
}

const Grid = ({ gridData, colors, setColors, setOriginalColors, selectedCells, onCellClick }) => {  //takes in the 2d list from model.py
  // const [colors, setColors] = useState([]); //colors stores our generated colours. initializes empty array and returns [current state val, function to update it]

  const generateColors = useCallback((length, width) => {
    const main_hues = [
        [255, 0, 0],    // Red
        [0, 0, 255],    // Blue
        [255, 255, 0],  // Yellow
        [255, 0, 255],  // Magenta
    ];
    const shuffled_hues = main_hues.sort(() => 0.5 - Math.random()).slice(0, 4);
    
    const newColors = Array(length * width).fill('');

    // Assign colors to the four fixed corners
    const cornerIndices = [
        0,                           // Top-left
        length - 1,                  // Top-right
        (length * width) - length,    // Bottom-left
        (length * width) - 1          // Bottom-right
    ];

    newColors[cornerIndices[0]] = `rgb(${shuffled_hues[0].join(',')})`;
    newColors[cornerIndices[1]] = `rgb(${shuffled_hues[1].join(',')})`;
    newColors[cornerIndices[2]] = `rgb(${shuffled_hues[2].join(',')})`;
    newColors[cornerIndices[3]] = `rgb(${shuffled_hues[3].join(',')})`;

    const interpolateColor = (color1, color2, factor) => {
        const [r1, g1, b1] = color1.slice(4, -1).split(',').map(Number);
        const [r2, g2, b2] = color2.slice(4, -1).split(',').map(Number);

        const r = Math.round(r1 + factor * (r2 - r1));
        const g = Math.round(g1 + factor * (g2 - g1));
        const b = Math.round(b1 + factor * (b2 - b1));

        return `rgb(${r},${g},${b})`;
    };

    // Assign colors to the top and bottom rows based on interpolation
    for (let i = 1; i < length - 1; i++) {
        newColors[i] = interpolateColor(newColors[cornerIndices[0]], newColors[cornerIndices[1]], i / (length - 1));
    }

    for (let i = (length * width) - length + 1; i < (length * width) - 1; i++) {
        newColors[i] = interpolateColor(newColors[cornerIndices[2]], newColors[cornerIndices[3]], (i - ((length * width) - length)) / (length - 1));
    }

    for (let i = 0; i < length; i++) {
        for (let j = i + length; j < (length * width) - length + i; j += length) {
            newColors[j] = interpolateColor(newColors[i], newColors[(length * width) - length + i], (j / length) / (width - 1));
        }
    }

    // Identify unmovable cells
    const unmovable = [...cornerIndices];  // Fixed corners should never move
    for (let i = 0; i < width; i++) {
        for (let j = 0; j < length; j++) {
            if (gridData[i][j] === 0) {
                unmovable.push(i * length + j);
            }
        }
    }

    // Create an array of indices from 0 to n-1
    const indices = Array.from(Array(length * width).keys());

    // Remove unmovable indices from the list of shuffle-able indices
    const movableIndices = indices.filter(i => !unmovable.includes(i));

    // Extract the colors corresponding to the movable indices
    const movableColors = movableIndices.map(i => newColors[i]);

    // Shuffle only the movable colors
    for (let i = movableColors.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [movableColors[i], movableColors[j]] = [movableColors[j], movableColors[i]];
    }

    // Assign the shuffled colors back to the original color matrix
    const jumbledColors = [...newColors];
    movableIndices.forEach((index, i) => {
        jumbledColors[index] = movableColors[i];
    });

    return { original: newColors, jumbled: jumbledColors }; 
  }, [gridData]);


  useEffect(() => {
    if (gridData && gridData.length > 0 && colors.length === 0) {
      const { original, jumbled } = generateColors(gridData[0].length, gridData.length);
      setColors(jumbled);
      setOriginalColors(original);
    }
  }, [gridData, colors, generateColors, setColors, setOriginalColors]); //when gridData changes, generateColours sets the gridData colours variable to a new set of colours

  if (!gridData || gridData.length === 0) { //error checking
    return <div>No grid data provided</div>;
  }

  const gridStyle = { //css creates the grid layout. # of cols is set dynamically based on input data
    display: 'grid',
    gridTemplateColumns: `repeat(${gridData[0].length}, 50px)`,
    gap: '2px',
    position: 'relative'
  };

  const squareStyle = { //css style for each square
    width: '50px',
    height: '50px',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    fontSize: '20px',
    color: 'white',
    position: 'relative',
    cursor: 'pointer'
  };

  const dotStyle = {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: '10px',
    height: '10px',
    borderRadius: '50%',
    backgroundColor: 'black',
  };

  return (
    <div style={gridStyle}>
      {gridData.flat().map((cell, index) => (
        <div
          key={index}
          style={{
            ...squareStyle,
            backgroundColor: colors[index] || 'gray',
            border: selectedCells.includes(index) ? '2px solid white' : 'none',
          }}
          onClick={() => onCellClick(index)}
        >
          {cell === 0 && <div style={dotStyle} />}
        </div>
      ))}
    </div>
  );
};


function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/mainmenu" element={<MainMenu />} />
        <Route path="/easylevel" element={<EasyLevel />} />
        <Route path="/mediumlevel" element={<MediumLevel />} />
        <Route path="/hardlevel" element={<HardLevel />} />
      </Routes>
    </Router>
  );
}

export default App;
