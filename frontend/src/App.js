import React, { useState, useEffect, useCallback, useMemo } from "react";
import "./App.css";
import axios from "axios";
import {
  Anchor, Navigation, BarChart3, CloudSun, Brain,
  Gauge, Fuel, Timer, Wind, Compass, Waves,
  Thermometer, Ship, MapPin, Zap, Eye, Droplets,
  TrendingUp, AlertTriangle, CheckCircle, XCircle, Activity, Sun, Gauge as GaugeIcon
} from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "";
const API = `${BACKEND_URL}/api`;

const EnhancedShipOptimizer = () => {
  const [optimizationData, setOptimizationData] = useState({
    departure_port: "Khalifa Port",
    arrival_port: "Ruwais Port",
    required_arrival_time: "",
    wind_speed: 5.0,
    wind_direction: 90.0
  });

  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [weatherData, setWeatherData] = useState(null);
  const [weatherLoading, setWeatherLoading] = useState(false);
  const [mlModelStatus, setMlModelStatus] = useState(null);
  const [activeTab, setActiveTab] = useState("optimization");
  const [currentTime, setCurrentTime] = useState(new Date());

  // Live Dubai clock
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatDubaiTime = (date) =>
    date.toLocaleString("en-GB", {
      timeZone: "Asia/Dubai",
      day: "2-digit", month: "short", year: "numeric",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
      hour12: false
    });

  // Default ETA: 12 hours from now in Dubai time
  useEffect(() => {
    const dubaiNow = new Date(new Date().toLocaleString("en-US", { timeZone: "Asia/Dubai" }));
    dubaiNow.setHours(dubaiNow.getHours() + 12);
    dubaiNow.setMinutes(0);
    dubaiNow.setSeconds(0);
    const y = dubaiNow.getFullYear();
    const m = String(dubaiNow.getMonth() + 1).padStart(2, "0");
    const d = String(dubaiNow.getDate()).padStart(2, "0");
    const h = String(dubaiNow.getHours()).padStart(2, "0");
    const min = String(dubaiNow.getMinutes()).padStart(2, "0");
    setOptimizationData(prev => ({ ...prev, required_arrival_time: `${y}-${m}-${d}T${h}:${min}` }));
  }, []);

  const fetchWeatherData = useCallback(async (departure, arrival) => {
    setWeatherLoading(true);
    try {
      const response = await axios.get(`${API}/weather`, {
        params: { departure_port: departure, arrival_port: arrival }
      });
      setWeatherData(response.data);
      if (response.data?.average) {
        setOptimizationData(prev => ({
          ...prev,
          wind_speed: Math.round((response.data.average.wind_speed || 0) * 10) / 10,
          wind_direction: Math.round(response.data.average.wind_direction || 0)
        }));
      }
    } catch {
      // Weather fetch failed — UI will fall back to defaults; no console noise in prod
    } finally {
      setWeatherLoading(false);
    }
  }, []);

  const fetchMlModelStatus = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/model-status`);
      setMlModelStatus(response.data);
    } catch {
      // ML status fetch failed — non-fatal
    }
  }, []);

  useEffect(() => {
    fetchWeatherData(optimizationData.departure_port, optimizationData.arrival_port);
    fetchMlModelStatus();
  }, [optimizationData.departure_port, optimizationData.arrival_port, fetchWeatherData, fetchMlModelStatus]);

  const handleOptimize = async () => {
    setIsLoading(true);
    setError("");
    setResults(null);

    if (optimizationData.departure_port === optimizationData.arrival_port) {
      setError("Departure and arrival ports cannot be the same.");
      setIsLoading(false);
      return;
    }

    try {
      const response = await axios.post(`${API}/optimize`, {
        ...optimizationData,
        priority_weights: { fuel: 0.6, time: 0.4 },
        request_alternatives: true
      });
      setResults(response.data);
      setActiveTab("results");
    } catch (err) {
      setError(err.response?.data?.detail || "Optimization failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputChange = (field, value) =>
    setOptimizationData(prev => ({ ...prev, [field]: value }));

  const tabs = [
    { id: "optimization", label: "Optimization", icon: Navigation },
    { id: "results", label: "Results", icon: BarChart3 },
    { id: "weather", label: "Weather", icon: CloudSun },
    { id: "mlmodel", label: "ML Model", icon: Brain },
  ];

  const r2pct = (mlModelStatus?.model_metrics?.r2_score || 0) * 100;
  const trainingVoyages = mlModelStatus?.data_statistics?.total_voyages || 0;

  return (
    <div className="min-h-screen bg-navy-950 font-body text-slate-200">
      {/* Header */}
      <header data-testid="app-header" className="sticky top-0 z-50 bg-navy-950/90 backdrop-blur-md border-b border-navy-700">
        <div className="max-w-[1800px] mx-auto px-3 md:px-8 py-3 md:py-4 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 md:gap-3 min-w-0">
            <Ship className="w-6 h-6 md:w-7 md:h-7 text-cyan-400 shrink-0" />
            <div className="min-w-0">
              <h1 className="font-heading font-bold text-sm md:text-xl text-white tracking-tight uppercase truncate">
                M/V Al-bazm II
              </h1>
              <p className="text-[10px] md:text-xs text-slate-400 font-body tracking-wide truncate">
                Maritime Fuel Optimization System
              </p>
            </div>
          </div>
          <div data-testid="live-clock" className="flex items-center gap-1.5 md:gap-2 shrink-0">
            <div className="live-dot w-2 h-2 rounded-full bg-emerald-500" />
            <span className="hidden md:inline text-xs text-slate-400 font-mono">DUBAI UTC+4</span>
            <span className="font-mono text-cyan-400 text-[11px] md:text-base font-medium tracking-wider whitespace-nowrap">
              {formatDubaiTime(currentTime)}
            </span>
          </div>
        </div>
      </header>

      <div className="max-w-[1800px] mx-auto px-3 md:px-8 py-4 md:py-6">
        {/* Tabs */}
        <nav data-testid="tab-navigation" className="flex gap-1 mb-4 md:mb-6 bg-navy-900 border border-navy-700 rounded-sm p-1 overflow-x-auto whitespace-nowrap">
          {tabs.map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                data-testid={`tab-${tab.id}`}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-3 md:px-4 py-2 text-xs md:text-sm font-medium transition-colors rounded-sm shrink-0 ${
                  isActive
                    ? "bg-cyan-400/10 text-cyan-400 border-b-2 border-cyan-400"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </nav>

        {/* Error */}
        {error && (
          <div data-testid="error-message" className="flex items-start gap-3 p-4 mb-6 bg-red-500/10 border border-red-500/30 rounded-sm">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <p className="text-red-300 text-sm font-mono whitespace-pre-line">{error}</p>
          </div>
        )}

        {/* OPTIMIZATION TAB */}
        {activeTab === "optimization" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div data-testid="voyage-planning-card" className="lg:col-span-2 bg-navy-900/60 border border-navy-700 rounded-sm p-5 md:p-6">
              <h2 className="font-heading text-lg font-semibold text-white uppercase tracking-tight mb-5 flex items-center gap-2">
                <Anchor className="w-5 h-5 text-cyan-400" />
                Voyage Planning
              </h2>

              <div className="space-y-5">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="font-mono text-xs uppercase tracking-wider text-slate-400 mb-2 block">
                      Departure Port
                    </label>
                    <select
                      data-testid="departure-port-select"
                      value={optimizationData.departure_port}
                      onChange={(e) => handleInputChange("departure_port", e.target.value)}
                      className="w-full p-3 bg-navy-900 border border-navy-700 text-white rounded-sm font-mono text-sm focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 focus:outline-none"
                    >
                      <option value="Khalifa Port">Khalifa Port</option>
                      <option value="Ruwais Port">Ruwais Port</option>
                    </select>
                  </div>
                  <div>
                    <label className="font-mono text-xs uppercase tracking-wider text-slate-400 mb-2 block">
                      Arrival Port
                    </label>
                    <select
                      data-testid="arrival-port-select"
                      value={optimizationData.arrival_port}
                      onChange={(e) => handleInputChange("arrival_port", e.target.value)}
                      className="w-full p-3 bg-navy-900 border border-navy-700 text-white rounded-sm font-mono text-sm focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 focus:outline-none"
                    >
                      <option value="Ruwais Port">Ruwais Port</option>
                      <option value="Khalifa Port">Khalifa Port</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="font-mono text-xs uppercase tracking-wider text-slate-400 mb-2 block">
                    Required Arrival Time (Dubai, UTC+4)
                  </label>
                  <input
                    data-testid="arrival-time-input"
                    type="datetime-local"
                    value={optimizationData.required_arrival_time}
                    onChange={(e) => handleInputChange("required_arrival_time", e.target.value)}
                    className="w-full p-3 bg-navy-900 border border-navy-700 text-white rounded-sm font-mono text-sm focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 focus:outline-none"
                    required
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="font-mono text-xs uppercase tracking-wider text-slate-400 mb-2 block">
                      Wind Speed (m/s)
                    </label>
                    <input
                      data-testid="wind-speed-input"
                      type="number"
                      step="0.1"
                      value={optimizationData.wind_speed}
                      onChange={(e) => handleInputChange("wind_speed", parseFloat(e.target.value))}
                      className="w-full p-3 bg-navy-900 border border-navy-700 text-white rounded-sm font-mono text-sm focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="font-mono text-xs uppercase tracking-wider text-slate-400 mb-2 block">
                      Wind Direction (&deg;)
                    </label>
                    <input
                      data-testid="wind-direction-input"
                      type="number"
                      step="1"
                      value={optimizationData.wind_direction}
                      onChange={(e) => handleInputChange("wind_direction", parseFloat(e.target.value))}
                      className="w-full p-3 bg-navy-900 border border-navy-700 text-white rounded-sm font-mono text-sm focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 focus:outline-none"
                    />
                  </div>
                </div>

                <button
                  data-testid="optimize-button"
                  onClick={handleOptimize}
                  disabled={isLoading}
                  className={`w-full py-3 rounded-sm font-heading font-semibold uppercase tracking-wide text-sm transition-colors ${
                    isLoading
                      ? "bg-slate-700 text-slate-400 cursor-not-allowed"
                      : "bg-cyan-400 text-navy-950 hover:bg-cyan-500"
                  }`}
                >
                  {isLoading ? "COMPUTING OPTIMAL ROUTE..." : "OPTIMIZE ROUTE"}
                </button>
              </div>
            </div>

            {/* Quick Stats Sidebar */}
            <div className="space-y-4">
              <QuickStatCard icon={Ship} label="Vessel" value="M/V Al-bazm II" testId="stat-vessel" />
              <QuickStatCard icon={Gauge} label="Max Speed" value="12.0 kn" testId="stat-max-speed" />
              <QuickStatCard icon={Activity} label="Optimal RPM" value="115-145" testId="stat-rpm" />
              <QuickStatCard icon={Brain} label="ML Accuracy" value={`${r2pct.toFixed(1)}%`} testId="stat-accuracy" />
              <QuickStatCard icon={TrendingUp} label="Training Voyages" value={trainingVoyages} testId="stat-voyages" />
            </div>
          </div>
        )}

        {/* RESULTS TAB */}
        {activeTab === "results" && results && (
          <div className="space-y-6">
            {/* Feasibility banner */}
            {results.eta_feasibility && !results.eta_feasibility.feasible && (
              <div data-testid="eta-warning" className="flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-sm">
                <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                <div className="text-red-300 text-sm font-mono space-y-1">
                  <p className="font-semibold text-red-200">ETA deadline cannot be met!</p>
                  <p>Required speed: {results.eta_feasibility.required_speed_kn} knots</p>
                  <p>Maximum speed: {results.eta_feasibility.max_speed_kn} knots</p>
                  <p>Minimum time needed: {results.eta_feasibility.min_hours_needed} hours</p>
                  <p>Suggested ETA: {results.eta_feasibility.suggested_eta_iso?.slice(0, 16).replace("T", " ")}</p>
                </div>
              </div>
            )}

            {/* Speed profile banner */}
            {results.speed_profile && (() => {
              const sp = results.speed_profile;
              const rr = results.recommended_route;
              let label = "UNIFORM";
              let detail = `Uniform speed across ${sp.total_distance_nm} NM`;
              if (sp.mode === "variable") {
                label = "VARIABLE";
                const avg = (rr.total_distance_nm / rr.estimated_duration_hours).toFixed(1);
                detail = `Variable ${sp.min_speed_kn?.toFixed(1)}–${sp.max_speed_kn?.toFixed(1)} kn · avg ${avg} kn · longer segments slowed to save fuel`;
              } else if (sp.mode === "constant-max") {
                label = "CONSTANT MAX (CRITICAL)";
                detail = `Running at 12.0 kn throughout — ETA is at the edge of feasibility (need avg ${sp.required_avg_kn} kn)`;
              }
              const isCritical = sp.mode === "constant-max";
              const boxCls = isCritical ? "bg-amber-500/10 border-amber-500/30" : "bg-cyan-400/5 border-cyan-400/30";
              const iconCls = isCritical ? "text-amber-400" : "text-cyan-400";
              return (
                <div data-testid="speed-profile" className={`p-4 rounded-sm border ${boxCls}`}>
                  <div className="flex items-center gap-2 mb-1">
                    <Activity className={`w-4 h-4 ${iconCls}`} />
                    <h4 className="font-mono text-xs uppercase tracking-wider text-slate-300">
                      Speed Profile: {label}
                    </h4>
                  </div>
                  <p className="text-sm text-slate-300 font-mono">{detail}</p>
                </div>
              );
            })()}

            {/* Metrics Row */}
            <div data-testid="results-metrics" className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
              <MetricCard icon={MapPin} title="Total Distance" value={results.recommended_route.total_distance_nm} unit="NM" testId="metric-distance" />
              <MetricCard icon={Timer} title="Total Trip Time" value={results.recommended_route.estimated_duration_hours} unit="hours" testId="metric-duration" />
              <MetricCard icon={Fuel} title="Estimated Fuel" value={results.recommended_route.total_fuel_mt} unit="MT" testId="metric-fuel" />
              <MetricCard icon={Activity} title="CO₂ Emissions" value={results.recommended_route.co2_emissions_mt} unit="MT" testId="metric-co2" />
            </div>

            {/* Physics corrections breakdown */}
            {results.physics_corrections && (
              <div data-testid="physics-corrections" className="bg-navy-900/60 border border-navy-700 rounded-sm p-5">
                <h3 className="font-heading text-base font-semibold text-white uppercase tracking-tight mb-4 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-cyan-400" />
                  Physics Corrections Applied
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 md:gap-4">
                  <div className="bg-navy-800/50 border border-navy-700 rounded-sm p-3 md:p-4">
                    <p className="font-mono text-xs text-slate-400 uppercase tracking-wider mb-1">Wind Effect</p>
                    <p className="font-mono text-xl md:text-2xl text-cyan-400">
                      {((results.physics_corrections.wind_correction.multiplier - 1) * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-slate-500 font-mono mt-1 capitalize">
                      {results.physics_corrections.wind_correction.label} · {Math.abs(results.physics_corrections.wind_correction.headwind_component_ms).toFixed(1)} m/s
                    </p>
                  </div>
                  <div className="bg-navy-800/50 border border-navy-700 rounded-sm p-3 md:p-4">
                    <p className="font-mono text-xs text-slate-400 uppercase tracking-wider mb-1">Wave Effect</p>
                    <p className="font-mono text-xl md:text-2xl text-cyan-400">
                      {((results.physics_corrections.wave_correction.multiplier - 1) * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-slate-500 font-mono mt-1">
                      H<sub>s</sub> = {results.physics_corrections.wave_correction.wave_height_m} m
                    </p>
                  </div>
                  <div className="bg-navy-800/50 border border-navy-700 rounded-sm p-3 md:p-4">
                    <p className="font-mono text-xs text-slate-400 uppercase tracking-wider mb-1">Base → Final</p>
                    <p className="font-mono text-sm text-white">
                      {results.physics_corrections.base_fuel_mt} → <span className="text-cyan-400 text-lg">{results.physics_corrections.corrected_fuel_mt}</span> MT
                    </p>
                    <p className="text-xs text-slate-500 font-mono mt-1">
                      Total multiplier ×{results.physics_corrections.total_multiplier}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Route Header */}
            <div className="bg-navy-900/60 border border-navy-700 rounded-sm p-1">
              <div className="px-5 py-3 border-b border-navy-700 flex items-center justify-between">
                <h3 className="font-heading text-base font-semibold text-white uppercase tracking-tight flex items-center gap-2">
                  <Navigation className="w-4 h-4 text-cyan-400" />
                  {results.recommended_route.route_name}
                  {results.recommended_route.original_route_name && (
                    <span className="text-xs text-slate-500 font-mono normal-case ml-2">
                      ({results.recommended_route.original_route_name})
                    </span>
                  )}
                </h3>
                <span className="bg-navy-700 text-cyan-400 font-mono text-xs px-2 py-1 rounded-sm border border-navy-700 uppercase">
                  Recommended
                </span>
              </div>
            </div>

            {/* Map */}
            <div className="bg-navy-900/60 border border-navy-700 rounded-sm p-5">
              <h3 className="font-heading text-base font-semibold text-white uppercase tracking-tight mb-4 flex items-center gap-2">
                <Compass className="w-4 h-4 text-cyan-400" />
                Route Map
              </h3>
              <RouteMap waypoints={results.recommended_route.waypoints} />
            </div>

            {/* Waypoints */}
            <div className="bg-navy-900/60 border border-navy-700 rounded-sm p-5">
              <h3 className="font-heading text-base font-semibold text-white uppercase tracking-tight mb-4 flex items-center gap-2">
                <MapPin className="w-4 h-4 text-cyan-400" />
                Waypoint Details
              </h3>
              <RouteTable waypoints={results.recommended_route.waypoints} />
            </div>

            {/* Insights */}
            <div className="bg-navy-900/60 border border-navy-700 rounded-sm p-5">
              <h3 className="font-heading text-base font-semibold text-white uppercase tracking-tight mb-4 flex items-center gap-2">
                <Zap className="w-4 h-4 text-cyan-400" />
                Optimization Insights
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {Object.entries(results.optimization_insights || {}).map(([key, value]) => (
                  <div key={key} className={`p-3 rounded-sm border ${
                    key === 'ml_model_info' || key === 'model_confidence'
                      ? 'bg-emerald-500/5 border-emerald-500/20'
                      : 'bg-navy-800/50 border-navy-700'
                  }`}>
                    <h4 className="font-mono text-xs uppercase tracking-wider text-slate-400 mb-1">
                      {key.replace(/_/g, ' ')}
                    </h4>
                    <p className="text-sm text-slate-300">{value}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Alternatives */}
            {results.alternative_routes?.length > 0 && (
              <div className="bg-navy-900/60 border border-navy-700 rounded-sm p-5">
                <h3 className="font-heading text-base font-semibold text-white uppercase tracking-tight mb-4 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-cyan-400" />
                  Pareto-Efficient Alternatives
                </h3>
                <div className="border border-navy-700 rounded-sm overflow-x-auto">
                  <table data-testid="alternatives-table" className="min-w-full text-sm">
                    <thead>
                      <tr className="bg-navy-900">
                        <th className="px-3 md:px-4 py-3 text-left font-mono text-xs uppercase tracking-wider text-slate-400 whitespace-nowrap">Route</th>
                        <th className="px-3 md:px-4 py-3 text-left font-mono text-xs uppercase tracking-wider text-slate-400">Type</th>
                        <th className="px-3 md:px-4 py-3 text-left font-mono text-xs uppercase tracking-wider text-slate-400 whitespace-nowrap">Speed (kn)</th>
                        <th className="px-3 md:px-4 py-3 text-left font-mono text-xs uppercase tracking-wider text-slate-400 whitespace-nowrap">Fuel (MT)</th>
                        <th className="px-3 md:px-4 py-3 text-left font-mono text-xs uppercase tracking-wider text-slate-400 whitespace-nowrap">CO₂ (MT)</th>
                        <th className="px-3 md:px-4 py-3 text-left font-mono text-xs uppercase tracking-wider text-slate-400 whitespace-nowrap">Time (hrs)</th>
                        <th className="px-3 md:px-4 py-3 text-left font-mono text-xs uppercase tracking-wider text-slate-400">Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-navy-700 bg-cyan-400/5">
                        <td className="px-3 md:px-4 py-3 text-sm text-cyan-400 font-medium whitespace-nowrap">{results.recommended_route.route_name}</td>
                        <td className="px-3 md:px-4 py-3"><span className="bg-navy-700 text-cyan-400 font-mono text-xs px-2 py-0.5 rounded-sm uppercase">Optimal</span></td>
                        <td className="px-3 md:px-4 py-3 font-mono text-sm text-cyan-400 whitespace-nowrap">{results.recommended_route.avg_speed_kn?.toFixed(1) || (results.recommended_route.total_distance_nm / results.recommended_route.estimated_duration_hours).toFixed(1)}{results.speed_profile?.mode === "variable" ? " (var)" : ""}</td>
                        <td className="px-3 md:px-4 py-3 font-mono text-sm text-white">{results.recommended_route.total_fuel_mt.toFixed(2)}</td>
                        <td className="px-3 md:px-4 py-3 font-mono text-sm text-amber-400">{results.recommended_route.co2_emissions_mt?.toFixed(2) || "—"}</td>
                        <td className="px-3 md:px-4 py-3 font-mono text-sm text-white">{results.recommended_route.estimated_duration_hours.toFixed(1)}</td>
                        <td className="px-3 md:px-4 py-3 font-mono text-sm text-cyan-400">{results.recommended_route.optimization_score.toFixed(3)}</td>
                      </tr>
                      {results.alternative_routes.map((route) => (
                        <tr key={route.route_id} className="border-b border-navy-700 hover:bg-navy-800/50 transition-colors">
                          <td className="px-3 md:px-4 py-3 text-sm text-slate-300 whitespace-nowrap">{route.route_name}</td>
                          <td className="px-3 md:px-4 py-3"><span className="bg-navy-700 text-slate-400 font-mono text-xs px-2 py-0.5 rounded-sm uppercase">{route.route_type}</span></td>
                          <td className="px-3 md:px-4 py-3 font-mono text-sm text-slate-300">{route.avg_speed_kn?.toFixed(1) ?? "—"}</td>
                          <td className="px-3 md:px-4 py-3 font-mono text-sm text-slate-300">{route.total_fuel_mt.toFixed(2)}</td>
                          <td className="px-3 md:px-4 py-3 font-mono text-sm text-amber-400/80">{route.co2_emissions_mt?.toFixed(2) || "—"}</td>
                          <td className="px-3 md:px-4 py-3 font-mono text-sm text-slate-300">{route.estimated_duration_hours.toFixed(1)}</td>
                          <td className="px-3 md:px-4 py-3 font-mono text-sm text-slate-400">{route.optimization_score.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "results" && !results && (
          <div className="bg-navy-900/60 border border-navy-700 rounded-sm p-12 text-center">
            <Navigation className="w-10 h-10 text-slate-600 mx-auto mb-3" />
            <p className="text-slate-500 font-mono text-sm">No optimization results yet. Run an optimization first.</p>
          </div>
        )}

        {/* WEATHER TAB */}
        {activeTab === "weather" && (
          <WeatherDisplay
            weatherData={weatherData}
            loading={weatherLoading}
            onRefresh={() => fetchWeatherData(optimizationData.departure_port, optimizationData.arrival_port)}
          />
        )}

        {/* ML MODEL TAB */}
        {activeTab === "mlmodel" && (
          <MlModelStatusDisplay mlModelStatus={mlModelStatus} />
        )}
      </div>

      <footer className="max-w-[1800px] mx-auto px-4 md:px-8 py-6 text-center text-xs text-slate-600 font-mono">
        M/V Al-bazm II Maritime Fuel Optimization &middot; Weather: Open-Meteo (live) &middot; ML: RandomForestRegressor
      </footer>
    </div>
  );
};

/* ---------- sub-components ---------- */

const QuickStatCard = ({ icon: Icon, label, value, testId }) => (
  <div data-testid={testId} className="metric-card bg-navy-900/60 border border-navy-700 rounded-sm p-4 flex items-center gap-3">
    <Icon className="w-5 h-5 text-cyan-400 shrink-0" />
    <div>
      <p className="font-mono text-xs text-slate-400 uppercase tracking-wider">{label}</p>
      <p className="font-mono text-base text-white font-medium">{value}</p>
    </div>
  </div>
);

const MetricCard = ({ icon: Icon, title, value, unit, testId }) => (
  <div data-testid={testId} className="metric-card bg-navy-900/60 border border-navy-700 rounded-sm p-5">
    <div className="flex items-center gap-2 mb-2">
      <Icon className="w-4 h-4 text-cyan-400" />
      <p className="font-mono text-xs text-slate-400 uppercase tracking-wider">{title}</p>
    </div>
    <p className="font-mono text-3xl text-cyan-400 font-medium">
      {typeof value === 'number' ? value.toFixed(value < 10 ? 2 : 1) : value}
    </p>
    {unit && <p className="font-mono text-xs text-slate-500 mt-1">{unit}</p>}
  </div>
);

const RouteMap = ({ waypoints }) => {
  useEffect(() => {
    if (typeof window === "undefined" || !window.L || !waypoints?.length) return undefined;
    const mapEl = document.getElementById("route-map");
    if (!mapEl) return undefined;
    // Clear any previous map nodes safely (no innerHTML assignment)
    while (mapEl.firstChild) mapEl.removeChild(mapEl.firstChild);
    const map = window.L.map("route-map", { zoomControl: true }).setView([24.5, 53.7], 9);
    window.L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 19
    }).addTo(map);

    const coords = waypoints.map(wp => [wp.lat, wp.lon]);
    const line = window.L.polyline(coords, { color: "#00F0FF", weight: 3, opacity: 0.8 }).addTo(map);

    waypoints.forEach((wp, i) => {
      const isEnd = i === 0 || i === waypoints.length - 1;
      // Use a DOM element instead of an html string for the marker icon (safer)
      const dot = document.createElement("div");
      const size = isEnd ? 14 : 10;
      Object.assign(dot.style, {
        width: `${size}px`,
        height: `${size}px`,
        borderRadius: "50%",
        background: isEnd ? "#FF5C00" : "#00F0FF",
        border: `2px solid ${isEnd ? "#fff" : "#0C1524"}`,
        boxShadow: isEnd ? "0 0 8px rgba(255,92,0,0.5)" : "0 0 8px rgba(0,240,255,0.4)",
      });
      const icon = window.L.divIcon({
        className: "",
        html: dot.outerHTML, // outerHTML of a programmatic element we built — safe
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2],
      });
      const marker = window.L.marker([wp.lat, wp.lon], { icon }).addTo(map);
      // Use Leaflet's text-only popup binding (escapes content) rather than HTML
      marker.bindPopup(`${wp.name} — Speed: ${wp.suggested_speed_kn} kn`);
    });

    map.fitBounds(line.getBounds(), { padding: [30, 30] });
    return () => map.remove();
  }, [waypoints]);

  return <div id="route-map" className="h-80 w-full" />;
};

const RouteTable = ({ waypoints }) => (
  <div className="border border-navy-700 rounded-sm overflow-x-auto">
    <table data-testid="waypoint-table" className="min-w-full text-sm">
      <thead>
        <tr className="bg-navy-900">
          <th className="px-3 md:px-4 py-3 text-left font-mono text-xs uppercase tracking-wider text-slate-400 whitespace-nowrap">Waypoint</th>
          <th className="px-3 md:px-4 py-3 text-left font-mono text-xs uppercase tracking-wider text-slate-400 whitespace-nowrap">Lat / Lon</th>
          <th className="px-3 md:px-4 py-3 text-left font-mono text-xs uppercase tracking-wider text-slate-400 whitespace-nowrap">Course</th>
          <th className="px-3 md:px-4 py-3 text-left font-mono text-xs uppercase tracking-wider text-slate-400 whitespace-nowrap">Distance (NM)</th>
          <th className="px-3 md:px-4 py-3 text-left font-mono text-xs uppercase tracking-wider text-slate-400 whitespace-nowrap">Speed (kn)</th>
        </tr>
      </thead>
      <tbody>
        {waypoints.map((wp, i) => {
          const spd = Number(wp.suggested_speed_kn) || 0;
          const ratio = Math.max(0, Math.min(1, (spd - 6) / 6));
          let speedColor = "text-emerald-400";
          if (ratio > 0.85) speedColor = "text-signal-orange";
          else if (ratio > 0.55) speedColor = "text-cyan-400";
          const key = `${wp.name}-${wp.lat}-${wp.lon}`;
          return (
            <tr key={key} className="border-b border-navy-700 hover:bg-navy-800/50 transition-colors">
              <td className="px-3 md:px-4 py-3 text-sm text-white font-medium whitespace-nowrap">{wp.name}</td>
              <td className="px-3 md:px-4 py-3 font-mono text-xs text-slate-400 whitespace-nowrap">{Number(wp.lat).toFixed(4)}, {Number(wp.lon).toFixed(4)}</td>
              <td className="px-3 md:px-4 py-3 font-mono text-sm text-slate-300 whitespace-nowrap">{wp.course_to_next ? Number(wp.course_to_next).toFixed(1) + "°" : "—"}</td>
              <td className="px-3 md:px-4 py-3 font-mono text-sm text-slate-300 whitespace-nowrap">{wp.distance_to_next_nm ? Number(wp.distance_to_next_nm).toFixed(2) : "—"}</td>
              <td className={`px-3 md:px-4 py-3 font-mono text-sm font-medium whitespace-nowrap ${speedColor}`}>{spd.toFixed(1)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  </div>
);

/* WEATHER TAB — uses LIVE Open-Meteo data */
const WeatherDisplay = ({ weatherData, loading, onRefresh }) => {
  if (loading || !weatherData) {
    return (
      <div className="bg-navy-900/60 border border-navy-700 rounded-sm p-12 text-center">
        <CloudSun className="w-10 h-10 text-slate-600 mx-auto mb-3 animate-pulse" />
        <p className="text-slate-500 font-mono text-sm">Fetching live marine weather…</p>
      </div>
    );
  }

  if (weatherData.success === false) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-sm p-6">
        <div className="flex items-center gap-3 mb-2">
          <AlertTriangle className="w-5 h-5 text-red-400" />
          <h3 className="text-red-300 font-mono">Weather service unavailable</h3>
        </div>
        <p className="text-red-200 text-sm">{weatherData.error || "Unable to fetch weather data."}</p>
      </div>
    );
  }

  const locations = [
    { key: "departure", label: "Departure" },
    { key: "midpoint",  label: "Midpoint" },
    { key: "arrival",   label: "Arrival" }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-heading text-lg font-semibold text-white uppercase tracking-tight flex items-center gap-2">
            <CloudSun className="w-5 h-5 text-cyan-400" />
            Live Marine Weather
          </h2>
          <p className="text-xs text-slate-500 font-mono mt-1">
            Source: Open-Meteo &middot; Updated: {weatherData.fetched_at?.replace("T", " ").slice(0, 19) || "now"}
          </p>
        </div>
        <button
          data-testid="refresh-weather"
          onClick={onRefresh}
          className="bg-cyan-400/10 text-cyan-400 hover:bg-cyan-400/20 px-4 py-2 rounded-sm text-xs font-mono uppercase tracking-wider border border-cyan-400/30"
        >
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {locations.map(({ key, label }) => {
          const d = weatherData[key];
          if (!d) return null;
          return (
            <div key={key} data-testid={`weather-${key}`} className="bg-navy-900/60 border border-navy-700 rounded-sm p-5">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <p className="text-xs text-slate-500 font-mono uppercase tracking-wider mb-1">{label}</p>
                  <h3 className="font-heading text-base font-semibold text-white uppercase tracking-tight">
                    {d.location_name || label}
                  </h3>
                </div>
                <Sun className="w-5 h-5 text-cyan-400" />
              </div>
              <div className="space-y-3">
                <WeatherRow icon={Thermometer} label="Air Temperature"
                  value={d.temperature !== null && d.temperature !== undefined ? `${d.temperature.toFixed(1)} °C` : "—"} />
                {d.apparent_temperature !== null && d.apparent_temperature !== undefined && (
                  <WeatherRow icon={Thermometer} label="Feels Like"
                    value={`${d.apparent_temperature.toFixed(1)} °C`} />
                )}
                <WeatherRow icon={Wind} label="Wind Speed"
                  value={d.wind_speed !== null && d.wind_speed !== undefined ? `${d.wind_speed.toFixed(1)} m/s` : "—"} />
                <WeatherRow icon={Compass} label="Wind Direction"
                  value={d.wind_direction !== null && d.wind_direction !== undefined ? `${Number(d.wind_direction).toFixed(0)}° ${degreesToCompass(d.wind_direction)}` : "—"} />
                {d.wind_gusts !== null && d.wind_gusts !== undefined && (
                  <WeatherRow icon={Wind} label="Wind Gusts"
                    value={`${d.wind_gusts.toFixed(1)} m/s`} />
                )}
                <WeatherRow icon={Droplets} label="Humidity"
                  value={d.humidity !== null && d.humidity !== undefined ? `${Math.round(d.humidity)}%` : "—"} />
                <WeatherRow icon={GaugeIcon} label="Pressure"
                  value={d.pressure !== null && d.pressure !== undefined ? `${Math.round(d.pressure)} hPa` : "—"} />
                <WeatherRow icon={Eye} label="Visibility"
                  value={d.visibility !== null && d.visibility !== undefined ? `${d.visibility} km` : "—"} />
                <WeatherRow icon={Waves} label="Wave Height"
                  value={d.wave_height !== null && d.wave_height !== undefined ? `${Number(d.wave_height).toFixed(2)} m` : "—"} />
                <WeatherRow icon={Waves} label="Sea State"
                  value={`Level ${d.sea_state ?? 0}`} />
                <WeatherRow icon={Activity} label="Impact Score"
                  value={d.impact_score?.toFixed(2)}
                  valueColor={impactScoreColor(d.impact_score)} />
              </div>
            </div>
          );
        })}
      </div>

      {/* Average + overall */}
      {weatherData.average && (
        <div className="bg-navy-900/60 border border-navy-700 rounded-sm p-5">
          <h3 className="font-heading text-base font-semibold text-white uppercase tracking-tight mb-4 flex items-center gap-2">
            <Wind className="w-4 h-4 text-cyan-400" />
            Route Average
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatBlock label="Avg Wind Speed" value={`${weatherData.average.wind_speed?.toFixed(1)} m/s`} color="text-cyan-400" />
            <StatBlock label="Avg Wind Direction" value={`${Math.round(weatherData.average.wind_direction)}° ${degreesToCompass(weatherData.average.wind_direction)}`} color="text-cyan-400" />
            <StatBlock label="Overall Impact" value={weatherData.overall_impact_score?.toFixed(2)} color="text-amber-400" />
          </div>
        </div>
      )}
    </div>
  );
};

const WeatherRow = ({ icon: Icon, label, value, valueColor = "text-white" }) => (
  <div className="flex items-center justify-between">
    <div className="flex items-center gap-2">
      <Icon className="w-3.5 h-3.5 text-slate-500" />
      <span className="text-sm text-slate-400">{label}</span>
    </div>
    <span className={`font-mono text-sm font-medium ${valueColor}`}>{value}</span>
  </div>
);

const degreesToCompass = (deg) => {
  if (deg === null || deg === undefined || isNaN(deg)) return "";
  const dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"];
  return dirs[Math.round(deg / 22.5) % 16];
};

const impactScoreColor = (score) => {
  if (score == null) return "text-white";
  if (score > 1.1) return "text-red-400";
  if (score < 0.9) return "text-emerald-400";
  return "text-amber-400";
};

const MlModelStatusDisplay = ({ mlModelStatus }) => {
  // Memoize the sorted feature-importance list so we don't re-sort on every render
  const topFeatures = useMemo(() => {
    const fi = mlModelStatus?.model_metrics?.feature_importance;
    if (!fi) return [];
    return Object.entries(fi)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 8);
  }, [mlModelStatus]);

  if (!mlModelStatus) {
    return (
      <div className="bg-navy-900/60 border border-navy-700 rounded-sm p-12 text-center">
        <Brain className="w-10 h-10 text-slate-600 mx-auto mb-3" />
        <p className="text-slate-500 font-mono text-sm">Loading ML model status…</p>
      </div>
    );
  }

  const r2 = mlModelStatus.model_metrics?.r2_score || 0;
  const mae = mlModelStatus.model_metrics?.mae || 0;
  const samples = mlModelStatus.model_metrics?.training_samples || 0;

  return (
    <div className="space-y-6">
      <div data-testid="ml-model-info" className="bg-navy-900/60 border border-navy-700 rounded-sm p-5">
        <h3 className="font-heading text-base font-semibold text-white uppercase tracking-tight mb-4 flex items-center gap-2">
          <Brain className="w-4 h-4 text-cyan-400" />
          Model Information
        </h3>
        <div className="space-y-3">
          <StatusRow label="Status" value={mlModelStatus.model_trained ? "Trained" : "Not Trained"}
            icon={mlModelStatus.model_trained ? CheckCircle : XCircle}
            iconColor={mlModelStatus.model_trained ? "text-emerald-400" : "text-red-400"} />
          <StatusRow label="Type" value={mlModelStatus.model_metrics?.model_type || "N/A"} />
          <StatusRow label="R² Accuracy" value={`${(r2 * 100).toFixed(1)}%`} valueColor="text-cyan-400" />
          <StatusRow label="MAE" value={`${mae.toFixed(3)} MT`} valueColor="text-amber-400" />
          <StatusRow label="Training Samples" value={samples} />
        </div>
      </div>

      {mlModelStatus.data_statistics && (
        <div className="bg-navy-900/60 border border-navy-700 rounded-sm p-5">
          <h3 className="font-heading text-base font-semibold text-white uppercase tracking-tight mb-4 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-cyan-400" />
            Performance Data
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatBlock label="Total Voyages" value={mlModelStatus.data_statistics.total_voyages} color="text-cyan-400" />
            <StatBlock label="Avg Fuel" value={`${mlModelStatus.data_statistics.mean_foc?.toFixed(1)} MT`} color="text-emerald-400" />
            <StatBlock label="Min Fuel" value={`${Number(mlModelStatus.data_statistics.min_foc).toFixed(2)} MT`} color="text-amber-400" />
            <StatBlock label="Max Fuel" value={`${Number(mlModelStatus.data_statistics.max_foc).toFixed(2)} MT`} color="text-signal-orange" />
          </div>
        </div>
      )}

      {topFeatures.length > 0 && (
        <div data-testid="feature-importance" className="bg-navy-900/60 border border-navy-700 rounded-sm p-5">
          <h3 className="font-heading text-base font-semibold text-white uppercase tracking-tight mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-cyan-400" />
            Feature Importance
          </h3>
          <div className="space-y-3">
            {topFeatures.map(([feature, importance]) => (
              <div key={feature} className="flex items-center gap-3">
                <span className="w-40 text-xs text-slate-400 font-mono uppercase truncate">{feature.replace(/_/g, ' ')}</span>
                <div className="flex-1 bg-navy-700 rounded-sm h-2 overflow-hidden">
                  <div className="feature-bar bg-gradient-to-r from-cyan-400 to-cyan-500 h-2 rounded-sm"
                    style={{ width: `${(importance * 100).toFixed(1)}%` }} />
                </div>
                <span className="font-mono text-xs text-cyan-400 w-12 text-right">{(importance * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-500 mt-4 font-mono">
            Higher importance = greater influence on fuel consumption prediction.
          </p>
        </div>
      )}
    </div>
  );
};

const StatusRow = ({ label, value, icon: Icon, iconColor, valueColor = "text-white" }) => (
  <div className="flex items-center justify-between">
    <span className="text-sm text-slate-400 capitalize">{label}</span>
    <div className="flex items-center gap-1.5">
      {Icon && <Icon className={`w-4 h-4 ${iconColor}`} />}
      <span className={`font-mono text-sm font-medium ${valueColor}`}>{value}</span>
    </div>
  </div>
);

const StatBlock = ({ label, value, color = "text-cyan-400" }) => (
  <div className="bg-navy-800/50 border border-navy-700 rounded-sm p-4">
    <p className="font-mono text-xs text-slate-400 uppercase tracking-wider mb-1">{label}</p>
    <p className={`font-mono text-2xl font-medium ${color}`}>{value}</p>
  </div>
);

const App = () => <EnhancedShipOptimizer />;
export default App;
