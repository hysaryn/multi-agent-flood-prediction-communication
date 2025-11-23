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
              <h1 className="text-2xl font-bold">Flood Risk Communication and Action Guidance System</h1>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
