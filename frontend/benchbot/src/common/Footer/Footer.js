import React from 'react';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <p>
          &copy; {new Date().getFullYear()} dbiz.ai. All rights reserved.
          <a 
            href="https://www.dbiz.ai" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            <strong> Click here</strong></a> to Visit our website.
          
        </p>
      </div>
    </footer>
  );
};

export default Footer;
