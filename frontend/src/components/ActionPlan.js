import React, { useState } from 'react';
import { 
  Shield, AlertCircle, Home, CheckCircle, FileText, ExternalLink, MapPin, 
  ChevronDown, ChevronUp, CreditCard, Package, Calendar, Lock, 
  LogOut, AlertTriangle, Camera, Eye, Wrench, Users, FileCheck, 
  MessageSquare, Layers, Heart, Trash2
} from 'lucide-react';
import mockActionPlanData from './MockActionPlan.json';

const ActionPlan = () => {
  const [activeSection, setActiveSection] = useState('before_flood');
  const [showSources, setShowSources] = useState(false);
  const [priorityFilter, setPriorityFilter] = useState('all');

  // Transform the JSON data structure to match component needs
  const transformActionPlanData = (data) => {
    if (!data || !data.final_plan) {
      return null;
    }

    const { final_plan, evaluation } = data;
    
    return {
      location: final_plan.location || 'Unknown Location',
      display_name: final_plan.display_name,
      before_flood: {
        title: 'Before Flood',
        icon: Shield,
        colorClass: 'blue',
        actions: final_plan.before_flood || []
      },
      during_flood: {
        title: 'During Flood',
        icon: AlertCircle,
        colorClass: 'red',
        actions: final_plan.during_flood || []
      },
      after_flood: {
        title: 'After Flood',
        icon: Home,
        colorClass: 'green',
        actions: final_plan.after_flood || []
      },
      sources: final_plan.sources || [],
      evaluation: evaluation || null
    };
  };

  const actionPlanData = transformActionPlanData(mockActionPlanData);
  
  if (!actionPlanData) {
    return (
      <div className="card compact-card action-plan-card">
        <p className="text-gray-600 text-sm">No action plan data available.</p>
      </div>
    );
  }

  const currentPlan = actionPlanData[activeSection];
  const Icon = currentPlan.icon;

  const getColorClasses = (colorClass) => {
    const colors = {
      blue: {
        icon: 'text-blue-600',
        border: 'border-blue-500',
        bg: 'bg-blue-50',
        text: 'text-blue-600',
        button: 'bg-blue-500',
        buttonHover: 'hover:bg-blue-600',
        priorityHigh: 'bg-red-100 text-red-700 border-red-300',
        priorityMedium: 'bg-yellow-100 text-yellow-700 border-yellow-300'
      },
      red: {
        icon: 'text-red-600',
        border: 'border-red-500',
        bg: 'bg-red-50',
        text: 'text-red-600',
        button: 'bg-red-500',
        buttonHover: 'hover:bg-red-600',
        priorityHigh: 'bg-red-100 text-red-700 border-red-300',
        priorityMedium: 'bg-yellow-100 text-yellow-700 border-yellow-300'
      },
      green: {
        icon: 'text-green-600',
        border: 'border-green-500',
        bg: 'bg-green-50',
        text: 'text-green-600',
        button: 'bg-green-500',
        buttonHover: 'hover:bg-green-600',
        priorityHigh: 'bg-red-100 text-red-700 border-red-300',
        priorityMedium: 'bg-yellow-100 text-yellow-700 border-yellow-300'
      }
    };
    return colors[colorClass] || colors.blue;
  };

  const getPriorityBadge = (priority) => {
    if (priority === 'high') {
      return (
        <span className="inline-flex items-center px-2.5 py-1 rounded text-sm font-medium bg-red-100 text-red-700 border border-red-300">
          High Priority
        </span>
      );
    } else if (priority === 'medium') {
      return (
        <span className="inline-flex items-center px-2.5 py-1 rounded text-sm font-medium bg-yellow-100 text-yellow-700 border border-yellow-300">
          Medium Priority
        </span>
      );
    }
    return null;
  };

  const getCategoryIcon = (category) => {
    const iconMap = {
      insurance: CreditCard,
      preparation: Package,
      planning: Calendar,
      protection: Lock,
      evacuation: LogOut,
      safety: AlertTriangle,
      documentation: Camera,
      recovery: Heart,
      cleanup: Trash2,
      maintenance: Wrench,
      awareness: Eye,
      compliance: FileCheck,
      information: FileText,
      communication: MessageSquare,
      'community contact': Users,
      community: Users,
      preparedness: Layers,
    };

    return iconMap[category] || CheckCircle;
  };

  const getCategoryBadge = (category) => {
    const categoryColors = {
      insurance: 'bg-blue-100 text-blue-700',
      preparation: 'bg-purple-100 text-purple-700',
      planning: 'bg-indigo-100 text-indigo-700',
      protection: 'bg-orange-100 text-orange-700',
      evacuation: 'bg-red-100 text-red-700',
      safety: 'bg-pink-100 text-pink-700',
      documentation: 'bg-gray-100 text-gray-700',
      recovery: 'bg-green-100 text-green-700',
      cleanup: 'bg-teal-100 text-teal-700',
      maintenance: 'bg-cyan-100 text-cyan-700',
      awareness: 'bg-amber-100 text-amber-700',
      compliance: 'bg-slate-100 text-slate-700',
      information: 'bg-sky-100 text-sky-700',
      communication: 'bg-violet-100 text-violet-700',
      community: 'bg-rose-100 text-rose-700',
      preparedness: 'bg-emerald-100 text-emerald-700'
    };

    const colorClass = categoryColors[category] || 'bg-gray-100 text-gray-700';
    return (
      <span className={`inline-flex items-center px-2.5 py-1 rounded text-sm font-medium ${colorClass}`}>
        {category}
      </span>
    );
  };

  const currentColors = getColorClasses(currentPlan.colorClass);

  const sections = [
    { key: 'before_flood', plan: actionPlanData.before_flood },
    { key: 'during_flood', plan: actionPlanData.during_flood },
    { key: 'after_flood', plan: actionPlanData.after_flood }
  ];

  // Filter actions by priority
  const filteredActions = currentPlan.actions && currentPlan.actions.length > 0
    ? currentPlan.actions.filter(action => 
        priorityFilter === 'all' || action.priority === priorityFilter
      )
    : [];

  return (
    <div className="card compact-card action-plan-card">
      {/* Sticky Header */}
      <div className="sticky top-0 bg-white z-10 pb-4 border-b border-gray-200 mb-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Icon className={`w-6 h-6 ${currentColors.icon}`} />
            <h2 className="text-xl font-bold">Flood Action Plan</h2>
          </div>
          {actionPlanData.location && (
            <div className="flex items-center gap-1 text-sm text-gray-600">
              <MapPin className="w-4 h-4" />
              <span>{actionPlanData.location}</span>
            </div>
          )}
        </div>

        {/* Section Tabs */}
        <div className="flex gap-1 mb-3">
          {sections.map(({ key, plan }) => {
            const PlanIcon = plan.icon;
            const isActive = activeSection === key;
            const planColors = getColorClasses(plan.colorClass);
            return (
              <button
                key={key}
                onClick={() => setActiveSection(key)}
                className={`flex items-center gap-1.5 px-3 py-2 text-sm font-medium transition-all rounded-lg ${
                  isActive
                    ? `${planColors.button} text-white shadow-sm`
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <PlanIcon className="w-4 h-4" />
                <span className="hidden sm:inline">{plan.title}</span>
                <span className="sm:hidden">{plan.title.split(' ')[0]}</span>
                {plan.actions && plan.actions.length > 0 && (
                  <span className={`ml-1 px-2 py-0.5 rounded text-sm ${
                    isActive ? 'bg-white/20 text-white' : 'bg-gray-200 text-gray-700'
                  }`}>
                    {plan.actions.length}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Priority Filter */}
        {currentPlan.actions && currentPlan.actions.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600 font-medium">Filter:</span>
            <div className="flex gap-1">
              {['all', 'high', 'medium'].map((filter) => (
                <button
                  key={filter}
                  onClick={() => setPriorityFilter(filter)}
                  className={`px-2.5 py-1 text-sm rounded transition-colors ${
                    priorityFilter === filter
                      ? 'bg-blue-100 text-blue-700 font-medium'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {filter === 'all' ? 'All' : filter.charAt(0).toUpperCase() + filter.slice(1)}
                </button>
              ))}
            </div>
            <span className="text-sm text-gray-500 ml-auto">
              Showing {filteredActions.length} of {currentPlan.actions.length} actions
            </span>
          </div>
        )}
      </div>

      {/* Scrollable Content Area */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Action Items - Scrollable */}
        <div className="action-plan-content space-y-3">
          {filteredActions.length > 0 ? (
            filteredActions.map((action, index) => {
              const CategoryIcon = getCategoryIcon(action.category);
              return (
              <div
                key={index}
                className={`border-l-4 ${currentColors.border} ${currentColors.bg} p-4 rounded-r-lg hover:shadow-md transition-all duration-200 hover:scale-[1.01]`}
              >
                <div className="flex items-start gap-3">
                  <div className={`${currentColors.button} text-white rounded-full p-2.5 flex-shrink-0 shadow-sm`}>
                    <CategoryIcon className="w-5 h-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <h3 className="text-base font-semibold text-gray-900 flex-1 leading-tight">
                        {action.title}
                      </h3>
                      {action.priority && (
                        <div className="flex-shrink-0">
                          {getPriorityBadge(action.priority)}
                        </div>
                      )}
                    </div>
                    <p className="text-sm text-gray-700 mb-3 leading-relaxed">
                      {action.description}
                    </p>
                    <div className="flex flex-wrap items-center gap-2">
                      {action.category && getCategoryBadge(action.category)}
                      {action.source_doc && (
                        <a
                          href={action.source_doc}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800 hover:underline font-medium"
                        >
                          <ExternalLink className="w-4 h-4" />
                          View Source
                        </a>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              );
            })
          ) : (
            <div className="text-center py-12 text-gray-500">
              <p className="text-base mb-1">No actions found</p>
              <p className="text-sm text-gray-400">
                {priorityFilter !== 'all' 
                  ? `Try changing the priority filter or check another section.`
                  : `No actions available for this section.`}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Fixed Footer Section */}
      <div className="flex-shrink-0 border-t border-gray-200 pt-4 mt-4 space-y-4">
        {/* Collapsible Sources Footer */}
        {actionPlanData.sources && actionPlanData.sources.length > 0 && (
          <div>
            <button
              onClick={() => setShowSources(!showSources)}
              className="w-full flex items-center justify-between p-2 text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
            >
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4" />
                <span className="text-sm font-semibold">Reference Sources ({actionPlanData.sources.length})</span>
              </div>
              {showSources ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>
            {showSources && (
              <div className="mt-2 p-3 bg-gray-50 rounded-lg border border-gray-200 max-h-48 overflow-y-auto">
                <ul className="space-y-2">
                  {actionPlanData.sources.map((source, index) => (
                    <li key={index} className="text-sm">
                      <a
                        href={source}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-800 hover:underline break-all flex items-start gap-2 group"
                      >
                        <ExternalLink className="w-4 h-4 flex-shrink-0 mt-0.5 group-hover:scale-110 transition-transform" />
                        <span className="flex-1">{source}</span>
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Quick Reference Footer */}
        <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex items-start gap-2 text-blue-800">
            <FileText className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <p className="text-sm leading-relaxed">
              <strong>Important:</strong> Always follow instructions from local emergency services. 
              This is a general guide - specific situations may require different actions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ActionPlan;

