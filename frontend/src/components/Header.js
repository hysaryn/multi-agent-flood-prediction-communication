import React from 'react';
import { Droplets } from 'lucide-react';

const Header = () => {
  return (
    <header className="bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          <div className="flex items-center space-x-3">
            <Droplets className="w-8 h-8" />
            <div>
              <h1 className="text-2xl font-bold">Flood Risk Communication</h1>
              <p className="text-sm text-blue-100">and Action Guidance System</p>
            </div>
          </div>
          <nav className="flex space-x-6">
            <a href="#" className="text-blue-100 hover:text-white transition-colors">About</a>
            <a href="#" className="text-blue-100 hover:text-white transition-colors">Data Sources</a>
            <a href="#" className="text-blue-100 hover:text-white transition-colors">Help</a>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;
