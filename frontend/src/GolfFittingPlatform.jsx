import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ScatterChart, Scatter, ResponsiveContainer } from 'recharts';
import { Settings, TrendingUp, Target, Wind } from 'lucide-react';

const GolfFittingPlatform = () => {
  const [swingParams, setSwingParams] = useState({
    clubheadSpeed: 95,
    attackAngle: 0,
    launchAngle: 12,
    spinRate: 2600,
    swingPath: 0
  });

  const [fittingResults, setFittingResults] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const handleAnalyzePy = async () => {
    setLoading(true);
    try {
      const payload = {
        clubhead_speed: swingParams.clubheadSpeed,
        attack_angle: swingParams.attackAngle,
        launch_angle: swingParams.launchAngle,
        spin_rate: swingParams.spinRate,
        swing_path: 0,
        face_angle: 0
      };
      const [fitRes, simRes] = await Promise.all([
        fetch('http://localhost:8000/fit', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        }),
        fetch('http://localhost:8000/simulate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            swing_params: payload,
            club_config: null
          })
        })
      ]);

      const fittingData = await fitRes.json();
      const baselineSim = await simRes.json();

      const optimizedPayload = {
        ...payload,
        launch_angle: payload.launch_angle + (fittingData.recommended_config.loft - 10.5),
        spin_rate: payload.spin_rate * 0.9
      }

      const optSimRes = await fetch('http://localhost:8000/simulate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          swing_params: optimizedPayload,
          club_config: fittingData.recommended_config
        })
      })

      const optimizedSim = await optSimRes.json();

      setFittingResults({
        // 1. Map backend metrics to frontend names
        baseline: {
          carryYards: baselineSim.metrics.carry_distance, // Backend already converted to yards
          totalYards: baselineSim.metrics.total_distance,
          apexYards: baselineSim.metrics.apex_height
        },
        optimized: {
          carryYards: optimizedSim.metrics.carry_distance,
          totalYards: optimizedSim.metrics.total_distance,
          apexYards: optimizedSim.metrics.apex_height
        },
        
        // 2. CONVERT TRAJECTORY TO YARDS
        // This ensures the line actually appears on a 0-250 scale
        baselineTrajectory: baselineSim.trajectory.map(p => ({
          x: p.x * 1.09361, // Convert meters to yards
          y: p.y * 1.09361  // Convert meters to yards
        })),
        optimizedTrajectory: optimizedSim.trajectory.map(p => ({
          x: p.x * 1.09361,
          y: p.y * 1.09361
        })),
      
        fitting: {
          loft: fittingData.recommended_config.loft,
          shaftFlex: fittingData.recommended_config.shaft_flex,
          shaftWeight: fittingData.recommended_config.shaft_weight,
          headWeight: fittingData.recommended_config.head_weight,
          confidence: fittingData.confidence_score
        },
        
        // Use the calculated difference for improvement
        improvement: optimizedSim.metrics.carry_distance - baselineSim.metrics.carry_distance,
        
        // Dispersion mapping (already using yards in your snippet)
        dispersion: Array.from({ length: 20 }, () => ({
          x: baselineSim.metrics.carry_distance + (Math.random() * 10 - 5),
          y: Math.random() * 10 - 5
        })),
        optimizedDispersion: Array.from({ length: 20 }, () => ({
          x: optimizedSim.metrics.carry_distance + (Math.random() * 8 - 4),
          y: Math.random() * 6 - 3
        }))
      });
    } catch (error) {
      console.error("Failed to fetch from API: ", error);
      alert("Make sure your Python FastAPI server is running on port 8000!");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center gap-3 mb-2">
            <Target className="w-8 h-8 text-green-600" />
            <h1 className="text-3xl font-bold text-gray-800">Virtual Golf Club Fitting Platform</h1>
          </div>
          <p className="text-gray-600">Physics-based driver optimization using ML-enhanced fitting algorithms</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Input Panel */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center gap-2 mb-4">
              <Settings className="w-5 h-5 text-green-600" />
              <h2 className="text-xl font-semibold">Swing Parameters</h2>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Clubhead Speed (mph): {swingParams.clubheadSpeed}
                </label>
                <input
                  type="range"
                  min="70"
                  max="130"
                  value={swingParams.clubheadSpeed}
                  onChange={(e) => setSwingParams({...swingParams, clubheadSpeed: Number(e.target.value)})}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Attack Angle (°): {swingParams.attackAngle}
                </label>
                <input
                  type="range"
                  min="-5"
                  max="5"
                  step="0.5"
                  value={swingParams.attackAngle}
                  onChange={(e) => setSwingParams({...swingParams, attackAngle: Number(e.target.value)})}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Launch Angle (°): {swingParams.launchAngle}
                </label>
                <input
                  type="range"
                  min="8"
                  max="18"
                  step="0.5"
                  value={swingParams.launchAngle}
                  onChange={(e) => setSwingParams({...swingParams, launchAngle: Number(e.target.value)})}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Spin Rate (rpm): {swingParams.spinRate}
                </label>
                <input
                  type="range"
                  min="1800"
                  max="4000"
                  step="100"
                  value={swingParams.spinRate}
                  onChange={(e) => setSwingParams({...swingParams, spinRate: Number(e.target.value)})}
                  className="w-full"
                />
              </div>

              <button
                onClick={handleAnalyzePy}
                disabled={loading}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors disabled:bg-gray-400"
              >
                {loading ? 'Analyzing...' : 'Analyze & Fit'}
              </button>
            </div>

            {fittingResults && (
              <div className="mt-6 pt-6 border-t border-gray-200">
                <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-green-600" />
                  Fitted Configuration
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Loft:</span>
                    <span className="font-medium">{fittingResults.fitting.loft}°</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Shaft Flex:</span>
                    <span className="font-medium">{fittingResults.fitting.shaftFlex}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Shaft Weight:</span>
                    <span className="font-medium">{fittingResults.fitting.shaftWeight}g</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Head Weight:</span>
                    <span className="font-medium">{fittingResults.fitting.headWeight}g</span>
                  </div>
                  <div className="flex justify-between pt-2 border-t">
                    <span className="text-gray-600">Confidence:</span>
                    <span className="font-bold text-green-600">{fittingResults.fitting.confidence}%</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2 space-y-6">
            {fittingResults && (
              <>
                {/* Performance Metrics */}
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <h3 className="text-xl font-semibold mb-4">Performance Comparison</h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center p-4 bg-gray-50 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Baseline Carry</div>
                      <div className="text-2xl font-bold text-gray-800">{fittingResults.baseline.carryYards.toFixed(2)} yds</div>
                    </div>
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Optimized Carry</div>
                      <div className="text-2xl font-bold text-green-600">{fittingResults.optimized.carryYards.toFixed(2)} yds</div>
                    </div>
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Improvement</div>
                      <div className="text-2xl font-bold text-blue-600">+{fittingResults.improvement.toFixed(1)} yds</div>
                    </div>
                  </div>
                </div>

                {/* Trajectory Chart */}
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <h3 className="text-xl font-semibold mb-4">Ball Flight Trajectory</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      {/* Explicitly set type="number" and domain to auto so it scales to ~250 yards */}
                      <XAxis 
                        dataKey="x" 
                        type="number" 
                        domain={[0, 'auto']} // This allows the axis to grow to 150+ yards
                        label={{ value: 'Distance (yards)', position: 'insideBottom', offset: -5 }} 
                      />
                      <YAxis 
                        type="number"
                        domain={[0, 'auto']} // This allows the axis to grow to the actual apex height
                        label={{ value: 'Height (yards)', angle: -90, position: 'insideLeft' }} 
                      />
                      <Tooltip />
                      <Legend />
                      
                      {/* Provide the data directly to each line */}
                      <Line 
                        data={fittingResults.baselineTrajectory} 
                        type="monotone" 
                        dataKey="y" 
                        stroke="#94a3b8" 
                        name="Baseline" 
                        dot={false} 
                        isAnimationActive={false} // Turn off animation to see if it renders immediately
                      />
                      <Line 
                        data={fittingResults.optimizedTrajectory} 
                        type="monotone" 
                        dataKey="y" 
                        stroke="#16a34a" 
                        name="Optimized" 
                        dot={false} 
                        strokeWidth={3}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Dispersion Pattern */}
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                    <Wind className="w-5 h-5 text-green-600" />
                    Shot Dispersion Pattern
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        type="number" 
                        dataKey="x" 
                        domain={['dataMin - 10', 'dataMax + 10']}
                        label={{ value: 'Carry Distance (yards)', position: 'insideBottom', offset: -5 }}
                      />
                      <YAxis 
                        type="number" 
                        dataKey="y"
                        domain={[-30, 30]}
                        label={{ value: 'Lateral Dispersion (yards)', angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Legend />
                      <Scatter 
                        name="Baseline" 
                        data={fittingResults.dispersion} 
                        fill="#6b7280" 
                        fillOpacity={0.4}
                      />
                      <Scatter 
                        name="Optimized" 
                        data={fittingResults.optimizedDispersion} 
                        fill="#10b981" 
                        fillOpacity={0.6}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}

            {!fittingResults && (
              <div className="bg-white rounded-lg shadow-lg p-12 text-center">
                <Target className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-800 mb-2">Ready to Optimize</h3>
                <p className="text-gray-600">
                  Enter your swing parameters and click "Analyze & Fit" to see your personalized driver configuration
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default GolfFittingPlatform;