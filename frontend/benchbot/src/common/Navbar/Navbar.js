import React from 'react';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-content">
        <img src="https://dbiz.ai/drupal/sites/default/files/preview-images/dbiz.ai-home.png" alt="dbiz logo" className="navbar-logo" />
        <h2 className="navbar-heading">BenchBot AI</h2>
        <div className="navbar-welcome">Welcome to BenchBot!</div>
      </div>
    </nav>
  );
};

export default Navbar;
