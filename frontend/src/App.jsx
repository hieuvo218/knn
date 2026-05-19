import React, { useState } from 'react';
import Home from './pages/Home.jsx';
import Model from './pages/Model.jsx';
import Database from './pages/Database.jsx';

const pages = {
  home: Home,
  model: Model,
  database: Database,
};

export default function App() {
  const [page, setPage] = useState('home');
  const Page = pages[page];

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>MNIST kNN</h1>
          <p>PostgreSQL source of truth · NPZ cache · kNN Flask ML</p>
        </div>
        <nav>
          <button className={page === 'home' ? 'active' : ''} onClick={() => setPage('home')}>Home</button>
          <button className={page === 'model' ? 'active' : ''} onClick={() => setPage('model')}>Model</button>
          <button className={page === 'database' ? 'active' : ''} onClick={() => setPage('database')}>Database</button>
        </nav>
      </header>
      <main>
        <Page />
      </main>
    </div>
  );
}
