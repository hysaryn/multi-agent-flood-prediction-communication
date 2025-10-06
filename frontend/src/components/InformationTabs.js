import React, { useState, useEffect } from 'react';
import OfficialGuide from './OfficialGuide';
import Live from './Live';
import SocialMedia from './SocialMedia';

const InformationTabs = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: 'OfficialGuide', label: 'Official Guide' },
    { id: 'Live', label: 'Live' },
    { id: 'SocialMedia', label: 'Social Media' }
  ];

  return (
    <div className="card">
      <div className="flex border-b">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`tab-button ${
              activeTab === tab.id ? 'active' : 'inactive'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      
      <div className="mt-4">
        {activeTab === 'OfficialGuide' && <OfficialGuide />}
        {activeTab === 'Live' && <Live />}
        {activeTab === 'SocialMedia' && <SocialMedia />}
      </div>
    </div>
  );
};

export default InformationTabs;
