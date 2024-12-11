import React from 'react';
import './HomePage.css';

const HomePage = ({ candidates, paginate, totalPages, currentPage }) => {
  return (
    <div className="homepage">
       {/* <h3 id="candidates">Available Bench Candidates</h3>
      <div>
        <ul>
          {candidates.length === 0 ? (
            <li>No candidates available at the moment.</li>
          ) : (
            candidates.map((candidate) => (
              <li key={candidate.Emp_ID}>
                {candidate.Emp_Name} - {candidate.Skillsets_Primary} - {candidate.Project_Name}
              </li>
            ))
          )}
        </ul>
      </div>  */}

      {/* Conditional rendering for pagination */}
      {/* {candidates.length > 0 && (
        <div className="pagination">
          {Array.from({ length: totalPages }, (_, index) => index + 1).map((pageNumber) => (
            <button
              key={pageNumber}
              onClick={() => paginate(pageNumber)}
              className={currentPage === pageNumber ? 'active' : ''}
            >
              {pageNumber}
            </button>
          ))}
        </div>
      )} */}

      {/* Chatbot Section */}
      <div id="chatbot" className="chatbot-container">
        {/* <h3>Chat with BenchBot:</h3> */}
        <iframe
          src="https://benchbot.onrender.com/" // backend chainlit URL
          // src="http://localhost:8000" // Chainlit URL
          width="100%"
          height="500px"
          title="BenchBot Chat"
        ></iframe>
      </div>
    </div>
  );
};

export default HomePage;
