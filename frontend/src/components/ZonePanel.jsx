import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { updateAlertStatus } from '../api.js';
import { useApp } from '../context/AppContext.jsx';

const RISK_COLORS = { Critical: '#C33A3A', High: '#C9622E', Medium: '#C98A1F', Low: '#3F8F5F' };

function ScoreBar({ label, value, max = 100 }) {
  const pct = Math.min(100, (value / max) * 100);
  const color = pct > 70 ? '#ef4444' : pct > 50 ? '#f97316' : pct > 30 ? '#eab308' : '#22c55e';
  return (
    <div className="factor-bar" style={{ marginBottom: 8 }}>
      <div className="factor-label">
        <span style={{ color: 'var(--text-secondary)', fontSize: '0.75rem' }}>{label}</span>
        <span style={{ fontWeight: 700, color, fontSize: '0.75rem' }}>{value?.toFixed(1)}</span>
      </div>
      <div className="factor-track" style={{ height: 6 }}>
        <div className="factor-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function ZonePanel({ data, onClose }) {
  const { score, history, sensor } = data;
  const { fetchAll, language, alerts, districts } = useApp();
  const color = RISK_COLORS[score.risk_level] || '#22c55e';
  const zoneAlerts = alerts.filter(a => a.grid_id === score.grid_id && a.status === 'Active');
  const district = districts.find(item => item.name === score.district || item.id === score.district);
  const districtName = district?.name || score.district;

  const chartData = (history || []).slice(0, 24).reverse().map((h, i) => ({
    t: `${i}`,
    score: h.composite_score,
    level: h.risk_level,
  }));

  const handleAck = async (alertId) => {
    await updateAlertStatus(alertId, 'Acknowledged', 'District Officer');
    fetchAll();
  };

  return (
    <div className="zone-panel">
      <h3>
        <span style={{ color }}>
          {score.grid_id} · {districtName}
        </span>
        <button className="close-btn" onClick={onClose} aria-label="Close selected zone">✕</button>
      </h3>

      {/* Risk gauge */}
      <div className="risk-gauge" style={{ marginBottom: 12 }}>
        <div className="gauge-score" style={{ color }}>{score.composite_score?.toFixed(1)}</div>
        <div className="gauge-label">Risk Level: {Math.round((score.composite_score || 0) / 10)}/10 · {score.risk_level}</div>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4 }}>
          Primary driver: <strong style={{ color: 'var(--text-secondary)' }}>{score.primary_factor}</strong>
        </div>
      </div>

      <div className="zone-facts"><span><b>24h rainfall</b>{score.rainfall_24h_mm ?? score.rainfall_score ?? '—'} mm</span><span><b>Primary factor</b>{score.primary_factor || '—'}</span><span><b>Incidents</b>{score.historical_incidents ?? score.historical_score ?? '—'}</span><span><b>Updated</b>{score.timestamp ? new Date(score.timestamp).toLocaleString() : '—'}</span></div>

      {/* Factor breakdown */}
      <ScoreBar label="🌧 Rainfall" value={score.rainfall_score} />
      <ScoreBar label="💧 Soil Moisture" value={score.soil_score} />
      <ScoreBar label="⛰ Slope Angle" value={score.slope_score} />
      <ScoreBar label="📚 Historical" value={score.historical_score} />
      <ScoreBar label="👥 Citizen Reports" value={score.citizen_score} />

      {/* Metadata */}
      <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginBottom: 10, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <span>🏔 {score.elevation}m</span>
        <span>📐 {score.slope_angle}°</span>
        <span>🛣 {score.road_proximity_km}km to road</span>
        <span>🏘 {score.village_proximity_km}km to village</span>
      </div>

      {/* Sensor data */}
      {sensor && sensor.length > 0 && (
        <div style={{ fontSize: '0.72rem', background: 'var(--bg-primary)', borderRadius: 8, padding: '8px 10px', marginBottom: 10 }}>
          <div style={{ color: 'var(--text-muted)', fontWeight: 700, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.08em', fontSize: '0.62rem' }}>Latest Sensor</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, color: 'var(--text-secondary)' }}>
            <span>🌧 {sensor[0].rainfall_1h_mm} mm/hr</span>
            <span>📅 {sensor[0].rainfall_24h_mm?.toFixed(0)} mm/24h</span>
            <span>💧 SM: {(sensor[0].soil_moisture * 100)?.toFixed(0)}%</span>
            <span>💧 Humidity: {sensor[0].humidity_pct?.toFixed(0)}%</span>
          </div>
        </div>
      )}

      <div className="shelter-box"><strong>Nearby shelters</strong><p>Shelter names and occupied/total capacity are not returned by the current districts endpoint.</p><small>Source status: unavailable in current API contract</small></div>

      {/* Risk trend chart */}
      {chartData.length > 1 && (
        <div className="chart-area">
          <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>Risk History (last 24 readings)</div>
          <ResponsiveContainer width="100%" height={80}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.4} />
                  <stop offset="95%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="2 4" stroke="var(--border)" />
              <XAxis dataKey="t" hide />
              <YAxis domain={[0, 100]} hide />
              <Tooltip
                contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.7rem' }}
                labelStyle={{ color: 'var(--text-muted)' }}
              />
              <Area type="monotone" dataKey="score" stroke={color} fill="url(#scoreGrad)" strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Active alerts for this zone */}
      {zoneAlerts.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>Active Alerts</div>
          {zoneAlerts.map(a => (
            <div key={a.id} style={{ background: 'var(--risk-critical-bg)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 8, padding: '8px 10px', marginBottom: 6 }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-primary)', marginBottom: 4 }}>
                {language === 'hi' ? a.message_hi : language === 'mz' ? a.message_mz : a.message_en}
              </div>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)' }}>🛣 {a.affected_roads} · 🏘 {a.affected_villages}</div>
              <button
                style={{ marginTop: 6, padding: '3px 10px', background: 'var(--border)', border: '1px solid var(--border-bright)', borderRadius: 6, color: 'var(--text-secondary)', fontSize: '0.68rem', cursor: 'pointer' }}
                onClick={() => handleAck(a.id)}
              >✓ Acknowledge</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
