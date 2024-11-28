import React from 'react';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <p>&copy; {new Date().getFullYear()} dbiz.ai. All rights reserved.
        <a href="https://www.dbiz.com">   Click here to Visit our website</a>
        </p>
      </div>
    </footer>
  );
};

export default Footer;
