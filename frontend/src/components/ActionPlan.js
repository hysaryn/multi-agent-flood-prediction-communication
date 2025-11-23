import React, { useState } from 'react';
import { Shield, AlertCircle, Home, CheckCircle, Clock, FileText } from 'lucide-react';

const ActionPlan = () => {
  const [activeSection, setActiveSection] = useState('pre');

  const actionPlans = {
    pre: {
      title: 'Pre-Flood Preparation',
      icon: Shield,
      colorClass: 'blue',
      actions: [
        {
          title: 'Create an Emergency Kit',
          description: 'Prepare a 72-hour emergency supply kit with essentials',
          items: [
            'Water (1 gallon per person per day)',
            'Non-perishable food items',
            'First aid kit and medications',
            'Flashlight and extra batteries',
            'Important documents in waterproof container',
            'Cash and credit cards',
            'Cell phone with charger',
            'Emergency contact information'
          ]
        },
        {
          title: 'Develop an Evacuation Plan',
          description: 'Plan your evacuation route and meeting points',
          items: [
            'Identify multiple evacuation routes',
            'Designate a meeting point for family members',
            'Plan for pets and livestock',
            'Know the location of emergency shelters',
            'Practice your evacuation plan with family',
            'Keep your vehicle fueled and ready'
          ]
        },
        {
          title: 'Protect Your Property',
          description: 'Take steps to minimize flood damage to your home',
          items: [
            'Install sump pumps with battery backup',
            'Elevate electrical panels and appliances',
            'Install check valves in sewer lines',
            'Waterproof your basement',
            'Clear gutters and drains',
            'Move valuable items to higher floors',
            'Consider flood insurance coverage'
          ]
        },
        {
          title: 'Stay Informed',
          description: 'Monitor weather conditions and flood warnings',
          items: [
            'Sign up for emergency alerts',
            'Monitor local news and weather reports',
            'Follow official social media accounts',
            'Download weather alert apps',
            'Know your flood risk level',
            'Understand flood warning terminology'
          ]
        }
      ]
    },
    during: {
      title: 'During Flood Event',
      icon: AlertCircle,
      colorClass: 'red',
      actions: [
        {
          title: 'Immediate Safety Actions',
          description: 'Take immediate steps to ensure your safety',
          items: [
            'Evacuate immediately if ordered to do so',
            'Move to higher ground if flooding begins',
            'Do not walk or drive through floodwaters',
            'Turn off electricity at the main breaker',
            'Avoid contact with floodwater (may be contaminated)',
            'Stay away from downed power lines',
            'Listen to emergency broadcasts'
          ]
        },
        {
          title: 'If Evacuating',
          description: 'Follow these steps when evacuating',
          items: [
            'Take your emergency kit',
            'Lock your home securely',
            'Follow designated evacuation routes',
            'Do not take shortcuts',
            'Avoid flooded roads and bridges',
            'Inform family of your destination',
            'Check in at emergency shelters if needed'
          ]
        },
        {
          title: 'If Staying Home',
          description: 'Safety measures if you must remain at home',
          items: [
            'Move to the highest level of your home',
            'Bring essential supplies with you',
            'Do not use electrical equipment in wet areas',
            'Avoid using tap water until authorities say it\'s safe',
            'Monitor water levels closely',
            'Be ready to evacuate if conditions worsen',
            'Keep emergency contacts accessible'
          ]
        },
        {
          title: 'Emergency Communication',
          description: 'Stay connected during the flood event',
          items: [
            'Keep phone charged and use sparingly',
            'Text instead of calling (uses less battery)',
            'Use social media to check in with family',
            'Listen to battery-powered radio for updates',
            'Follow official emergency services',
            'Report emergencies to 911',
            'Do not spread unverified information'
          ]
        }
      ]
    },
    after: {
      title: 'After Flood Event',
      icon: Home,
      colorClass: 'green',
      actions: [
        {
          title: 'Returning Home Safely',
          description: 'Important safety checks before re-entering your home',
          items: [
            'Wait for authorities to declare it safe to return',
            'Check for structural damage before entering',
            'Turn off electricity at main breaker if not already done',
            'Do not use electrical equipment if wet',
            'Check for gas leaks',
            'Inspect for damage to water and sewer lines',
            'Take photos of damage for insurance'
          ]
        },
        {
          title: 'Cleaning and Recovery',
          description: 'Steps to safely clean up after flooding',
          items: [
            'Wear protective clothing (gloves, boots, mask)',
            'Remove standing water and mud',
            'Discard contaminated food and water',
            'Clean and disinfect all surfaces',
            'Dry out your home thoroughly (prevent mold)',
            'Remove wet carpet and padding',
            'Clean and disinfect HVAC systems'
          ]
        },
        {
          title: 'Documentation and Insurance',
          description: 'Document damage for insurance claims',
          items: [
            'Take detailed photos and videos of damage',
            'Create an inventory of damaged items',
            'Keep receipts for cleanup and repair expenses',
            'Contact your insurance company immediately',
            'File a claim as soon as possible',
            'Keep records of all communications',
            'Do not dispose of damaged items until inspected'
          ]
        },
        {
          title: 'Health and Safety',
          description: 'Protect your health during recovery',
          items: [
            'Avoid contact with floodwater (may contain sewage)',
            'Wash hands frequently with soap and clean water',
            'Watch for signs of illness',
            'Ensure safe drinking water',
            'Be cautious of mold growth',
            'Seek medical attention if injured',
            'Take care of mental health - recovery takes time'
          ]
        }
      ]
    }
  };

  const currentPlan = actionPlans[activeSection];
  const Icon = currentPlan.icon;

  const getColorClasses = (colorClass) => {
    const colors = {
      blue: {
        icon: 'text-blue-600',
        border: 'border-blue-500',
        bg: 'bg-blue-50',
        text: 'text-blue-600',
        button: 'bg-blue-500',
        buttonHover: 'hover:bg-blue-600'
      },
      red: {
        icon: 'text-red-600',
        border: 'border-red-500',
        bg: 'bg-red-50',
        text: 'text-red-600',
        button: 'bg-red-500',
        buttonHover: 'hover:bg-red-600'
      },
      green: {
        icon: 'text-green-600',
        border: 'border-green-500',
        bg: 'bg-green-50',
        text: 'text-green-600',
        button: 'bg-green-500',
        buttonHover: 'hover:bg-green-600'
      }
    };
    return colors[colorClass] || colors.blue;
  };

  const currentColors = getColorClasses(currentPlan.colorClass);

  return (
    <div className="card compact-card action-plan-card">
      <div className="flex items-center gap-2 mb-4">
        <Icon className={`w-6 h-6 ${currentColors.icon}`} />
        <h2 className="text-xl font-bold">Flood Action Plan</h2>
      </div>

      {/* Section Tabs */}
      <div className="flex gap-1 mb-4 border-b border-gray-200">
        {Object.entries(actionPlans).map(([key, plan]) => {
          const PlanIcon = plan.icon;
          const isActive = activeSection === key;
          const planColors = getColorClasses(plan.colorClass);
          return (
            <button
              key={key}
              onClick={() => setActiveSection(key)}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium transition-all border-b-2 ${
                isActive
                  ? `${planColors.border} ${planColors.text} ${planColors.bg}`
                  : 'border-transparent text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              <PlanIcon className="w-4 h-4" />
              <span className="hidden sm:inline">{plan.title}</span>
              <span className="sm:hidden">{plan.title.split(' ')[0]}</span>
            </button>
          );
        })}
      </div>

      {/* Action Items - Scrollable */}
      <div className="action-plan-content space-y-4">
        {currentPlan.actions.map((action, index) => (
          <div
            key={index}
            className={`border-l-4 ${currentColors.border} ${currentColors.bg} p-3 rounded-r-lg`}
          >
            <div className="flex items-start gap-2 mb-2">
              <div className={`${currentColors.button} text-white rounded-full p-1.5 flex-shrink-0`}>
                <CheckCircle className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-semibold text-gray-900 mb-1">
                  {action.title}
                </h3>
                <p className="text-xs text-gray-600 mb-2">
                  {action.description}
                </p>
                <ul className="space-y-1">
                  {action.items.map((item, itemIndex) => (
                    <li key={itemIndex} className="flex items-start gap-1.5 text-xs text-gray-700">
                      <span className={`${currentColors.text} mt-0.5 flex-shrink-0`}>•</span>
                      <span className="flex-1">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Quick Reference Footer */}
      <div className="mt-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
        <div className="flex items-start gap-2 text-gray-600">
          <FileText className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <p className="text-xs">
            <strong>Remember:</strong> Always follow instructions from local emergency services. 
            This is a general guide - specific situations may require different actions.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ActionPlan;

