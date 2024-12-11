import React, { useState, useEffect } from 'react';
import './App.css';
import HomePage from './component/Homepage/HomePage';
import Navbar from './common/Navbar/Navbar';
import Footer from './common/Footer/Footer';
import axios from 'axios'; 


function App() {
  const [candidates, setCandidates] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const candidatesPerPage = 5;

  useEffect(() => {
    console.log("urll:",process.env.REACT_APP_BASE_URL)
    axios.get(`${process.env.REACT_BASE_URL}`) 
      .then(response => {
        const indexOfLastCandidate = currentPage * candidatesPerPage;
        const indexOfFirstCandidate = indexOfLastCandidate - candidatesPerPage;
        const currentCandidates = response.data.slice(indexOfFirstCandidate, indexOfLastCandidate);
        setCandidates(currentCandidates); 
      })
      .catch(error => console.error('Error fetching candidates:', error));
  }, [currentPage]); // Trigger effect when currentPage changes

  // Pagination buttons logic
  const paginate = (pageNumber) => setCurrentPage(pageNumber);

  // Calculate the total number of pages
  const totalPages = Math.ceil(50 / candidatesPerPage); // Assuming we have 50 candidates in total
  return (
    <div className="App">
      <Navbar />
      <HomePage candidates={candidates} paginate={paginate} totalPages={totalPages} currentPage={currentPage} />
      <Footer />
    </div>
  );
}

export default App;
