import { getDB } from '../db/database.js';
import { mlPredict, MlServiceError } from './mlClient.js';

// ---------------------------------------------------------------------------
// Sensor simulation — generates realistic monsoon-season readings
//
// NER REALISTIC RAINFALL RANGES (FIX-1.1):
//   Normal monsoon day : rainfall_1h_mm  2–18 mm/hr
//                        rainfall_24h_mm 15–120 mm/24h  (most days)
//   Heavy event (15%)  : rainfall_1h_mm  18–40 mm/hr
//                        rainfall_24h_mm 120–280 mm/24h  (genuine extremes)
//   Non-monsoon (Oct–May): rainfall much lower, 1–8 mm/hr
//
// KEY FIX: old code used rain24h = rain1h * (18–30) → produced 250–800mm/24h,
// saturating the risk normalizer (cap at 150mm) almost every monsoon cycle.
// New formula computes rain24h independently with a realistic spread so that
// baseline simulations produce a mix of Low / Medium / High / Critical zones.
// The spike demo path (POST /api/risk/trigger {spike:true}) is UNCHANGED.
// ---------------------------------------------------------------------------
export function simulateSensorFeed() {
  const db = getDB();
  const grids = db.prepare('SELECT * FROM grid_cells').all();
  const now = new Date().toISOString();

  const month = new Date().getMonth() + 1;
  const isMonsoon = month >= 6 && month <= 10;

  const insert = db.prepare(`
    INSERT INTO sensor_readings(grid_id,timestamp,rainfall_1h_mm,rainfall_24h_mm,soil_moisture,temperature_c,humidity_pct,source)
    VALUES (?,?,?,?,?,?,?,?)
  `);

  const readings = [];
  for (const g of grids) {
    const geoFactor = g.geology === 'fractured_gneiss' ? 1.3 : g.geology === 'schist' ? 1.15 : 1.0;
    const slopeFactor = g.slope_angle > 40 ? 1.15 : 1.0;

    // Determine if this reading represents a heavy-event cell (15% chance in monsoon, 5% otherwise)
    const isHeavyEvent = Math.random() < (isMonsoon ? 0.15 : 0.05);

    // rainfall_1h_mm: realistic hourly intensity
    // Normal monsoon: 2–18 mm/hr; Heavy event: 18–40 mm/hr; Non-monsoon: 0.5–6 mm/hr
    let rain1hBase;
    if (isHeavyEvent) {
      rain1hBase = 18 + Math.random() * 22; // 18–40 mm/hr
    } else if (isMonsoon) {
      rain1hBase = 2 + Math.random() * 16;  // 2–18 mm/hr
    } else {
      rain1hBase = 0.5 + Math.random() * 5.5; // 0.5–6 mm/hr
    }
    const rain1h = parseFloat((rain1hBase * geoFactor).toFixed(2));

    // rainfall_24h_mm: computed independently — NOT derived by multiplying rain1h × hours.
    // That old approach created unrealistic 24h totals by assuming peak 1h rate all day.
    // Instead use a realistic 24h accumulation range:
    //   Heavy event: 120–280 mm/24h (genuine extreme for NER)
    //   Normal monsoon: 15–120 mm/24h (typical NER monsoon day)
    //   Non-monsoon: 2–25 mm/24h
    let rain24hBase;
    if (isHeavyEvent) {
      rain24hBase = 120 + Math.random() * 160; // 120–280 mm/24h
    } else if (isMonsoon) {
      rain24hBase = 15 + Math.random() * 105;  // 15–120 mm/24h
    } else {
      rain24hBase = 2 + Math.random() * 23;    // 2–25 mm/24h
    }
    const rain24h = parseFloat((rain24hBase * slopeFactor).toFixed(2));

    // Soil moisture: driven by 24h accumulation, bounded 0.3–0.95
    const soilMoisture = parseFloat(Math.min(0.95, Math.max(0.30,
      0.35 + (rain24h / 280) * 0.55 + Math.random() * 0.08
    )).toFixed(3));

    const tempC = parseFloat((18 - g.elevation / 500 + Math.random() * 4).toFixed(1));
    const humidity = parseFloat(Math.min(100, (isMonsoon ? 70 : 50) + Math.random() * 25).toFixed(1));

    insert.run(g.id, now, rain1h, rain24h, soilMoisture, tempC, humidity, 'simulated');
    readings.push({ grid_id: g.id, rainfall_1h_mm: rain1h, rainfall_24h_mm: rain24h, soil_moisture: soilMoisture });
  }
  return readings;
}

// ---------------------------------------------------------------------------
// Rolling rainfall aggregation helper
// Returns { rainfall_3d_accum_mm, rainfall_7d_accum_mm } per grid cell
// ---------------------------------------------------------------------------
function getRollingRainfall(db, gridId) {
  const rows3d = db.prepare(`
    SELECT rainfall_24h_mm FROM sensor_readings
    WHERE grid_id = ?
    ORDER BY timestamp DESC
    LIMIT 3
  `).all(gridId);

  const rows7d = db.prepare(`
    SELECT rainfall_24h_mm FROM sensor_readings
    WHERE grid_id = ?
    ORDER BY timestamp DESC
    LIMIT 7
  `).all(gridId);

  const sum3d = rows3d.reduce((acc, r) => acc + (r.rainfall_24h_mm || 0), 0);
  const sum7d = rows7d.reduce((acc, r) => acc + (r.rainfall_24h_mm || 0), 0);

  return {
    rainfall_3d_accum_mm: parseFloat(sum3d.toFixed(2)),
    rainfall_7d_accum_mm: parseFloat(sum7d.toFixed(2)),
  };
}

// ---------------------------------------------------------------------------
// Feature vector builder — assembles all 16 features per grid cell
// ---------------------------------------------------------------------------
function buildFeatureVector(db, g, sensor, maxHistoricalIncidents) {
  const rolling = getRollingRainfall(db, g.id);

  // historical_landslide_density: normalized 0–1 (incidents / max across all cells)
  const historicalDensity = maxHistoricalIncidents > 0
    ? parseFloat((g.historical_incidents / maxHistoricalIncidents).toFixed(4))
    : 0;

  return {
    grid_id: g.id,

    // Terrain (static, DEM-derived estimates — labelled as estimated)
    elevation_m:        g.elevation,
    slope_deg:          g.slope_angle,
    aspect_deg:         g.aspect_deg         ?? 180.0,  // estimated
    plan_curvature:     g.plan_curvature      ?? -0.001, // estimated
    profile_curvature:  g.profile_curvature   ?? -0.001, // estimated
    twi:                g.twi                 ?? 10.0,   // estimated
    dist_to_road_m:     (g.road_proximity_km ?? 1.0) * 1000,  // km → m
    dist_to_stream_m:   g.dist_to_stream_m   ?? 300.0,  // estimated

    // Categorical (exact strings required by label encoders)
    land_cover_type:    g.land_cover_type     ?? 'Dense Forest',
    soil_type:          g.soil_type           ?? 'Weathered Gneiss & Mica Schist',

    // Dynamic / sensor
    rainfall_24h_mm:         sensor?.rainfall_24h_mm || 0,
    rainfall_intensity_mm_h: sensor?.rainfall_1h_mm  || 0,
    soil_moisture_pct:       parseFloat(((sensor?.soil_moisture || 0.3) * 100).toFixed(2)),

    // Rolling accumulation
    ...rolling,

    // Derived
    historical_landslide_density: historicalDensity,
  };
}

// ---------------------------------------------------------------------------
// Old rule-based scoring (kept as fallback when ml-service is unavailable)
// ---------------------------------------------------------------------------
function computeRuleBasedScores(db, grids) {
  const getCfg = (key, def) => {
    const row = db.prepare('SELECT value FROM admin_config WHERE key=?').get(key);
    return row ? parseFloat(row.value) : def;
  };

  const W = {
    rainfall:   getCfg('weight_rainfall', 0.35),
    soil:       getCfg('weight_soil', 0.25),
    slope:      getCfg('weight_slope', 0.20),
    historical: getCfg('weight_historical', 0.12),
    citizen:    getCfg('weight_citizen', 0.08),
  };

  const critical = getCfg('risk_threshold_critical', 75);
  const high     = getCfg('risk_threshold_high', 55);
  const medium   = getCfg('risk_threshold_medium', 30);

  const scores = [];
  for (const g of grids) {
    const sensor = db.prepare(
      'SELECT * FROM sensor_readings WHERE grid_id=? ORDER BY timestamp DESC LIMIT 1'
    ).get(g.id);

    const rain1h       = sensor?.rainfall_1h_mm  || 0;
    const rain24h      = sensor?.rainfall_24h_mm || 0;
    const soilMoisture = sensor?.soil_moisture   || 0.3;

    const rainfallScore  = Math.min(1.0, (rain1h / 50) * 0.4 + (rain24h / 150) * 0.6);
    const soilScore      = soilMoisture;
    const slopeScore     = Math.min(1.0, g.slope_angle / 45);
    const historicalScore = Math.min(1.0, g.historical_incidents / 12);

    const recentReports = db.prepare(
      `SELECT COUNT(*) as cnt FROM citizen_reports WHERE grid_id=? AND submitted_at > datetime('now', '-24 hours')`
    ).get(g.id);
    const citizenScore = Math.min(1.0, (recentReports?.cnt || 0) / 3);

    const composite = parseFloat((
      (W.rainfall * rainfallScore + W.soil * soilScore + W.slope * slopeScore +
       W.historical * historicalScore + W.citizen * citizenScore) * 100
    ).toFixed(2));

    let riskLevel;
    if      (composite >= critical) riskLevel = 'Critical';
    else if (composite >= high)     riskLevel = 'High';
    else if (composite >= medium)   riskLevel = 'Medium';
    else                             riskLevel = 'Low';

    const factors = [
      { name: 'Rainfall',            score: W.rainfall   * rainfallScore },
      { name: 'Soil Moisture',       score: W.soil       * soilScore },
      { name: 'Slope Angle',         score: W.slope      * slopeScore },
      { name: 'Historical Incidents',score: W.historical * historicalScore },
      { name: 'Citizen Reports',     score: W.citizen    * citizenScore },
    ];
    const primaryFactor = factors.sort((a, b) => b.score - a.score)[0].name;

    scores.push({
      grid_id:          g.id,
      rainfall_score:   parseFloat((rainfallScore  * 100).toFixed(2)),
      soil_score:       parseFloat((soilScore      * 100).toFixed(2)),
      slope_score:      parseFloat((slopeScore     * 100).toFixed(2)),
      historical_score: parseFloat((historicalScore* 100).toFixed(2)),
      citizen_score:    parseFloat((citizenScore   * 100).toFixed(2)),
      composite_score:  composite,
      risk_level:       riskLevel,
      primary_factor:   primaryFactor,
      ml_probability:   null,
      model_version:    'rule-based-fallback',
      degraded_mode:    true,
    });
  }
  return scores;
}

// ---------------------------------------------------------------------------
// Main risk scoring engine — ML-powered with rule-based fallback
// ---------------------------------------------------------------------------
export async function computeRiskScores() {
  const db = getDB();
  const grids = db.prepare('SELECT * FROM grid_cells').all();
  const now = new Date().toISOString();

  // Compute max historical incidents for density normalization
  const maxHistoricalIncidents = grids.reduce((max, g) => Math.max(max, g.historical_incidents || 0), 1);

  // Build feature vectors for all grids
  const featureVectors = grids.map(g => {
    const sensor = db.prepare(
      'SELECT * FROM sensor_readings WHERE grid_id=? ORDER BY timestamp DESC LIMIT 1'
    ).get(g.id);
    return buildFeatureVector(db, g, sensor, maxHistoricalIncidents);
  });

  let mlResults = null;
  let degradedMode = false;
  let modelVersion = 'ml-xgboost-v2';

  // --- Primary: call ml-service ---
  try {
    mlResults = await mlPredict(featureVectors);
    console.log(`[RiskEngine] ML inference completed for ${mlResults.length} grid cells`);
  } catch (err) {
    if (err instanceof MlServiceError) {
      console.warn(`[RiskEngine] ml-service unavailable — falling back to rule-based: ${err.message}`);
      degradedMode = true;
    } else {
      console.error('[RiskEngine] Unexpected error calling ml-service:', err);
      degradedMode = true;
    }
  }

  // --- Fallback: rule-based scoring ---
  if (degradedMode || !mlResults) {
    const fallbackScores = computeRuleBasedScores(db, grids);
    const insert = db.prepare(`
      INSERT INTO risk_scores(grid_id,timestamp,rainfall_score,soil_score,slope_score,historical_score,citizen_score,composite_score,risk_level,primary_factor,ml_probability,model_version)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    `);
    const scores = [];
    for (const s of fallbackScores) {
      insert.run(
        s.grid_id, now,
        s.rainfall_score, s.soil_score, s.slope_score, s.historical_score, s.citizen_score,
        s.composite_score, s.risk_level, s.primary_factor,
        null, 'rule-based-fallback'
      );
      const g = grids.find(x => x.id === s.grid_id);
      scores.push({
        ...s,
        timestamp: now,
        lat: g?.lat, lng: g?.lng,
        district: g?.district,
        slope_angle: g?.slope_angle,
        elevation: g?.elevation,
        road_proximity_km: g?.road_proximity_km,
        village_proximity_km: g?.village_proximity_km,
        degraded_mode: true,
      });
    }
    return scores;
  }

  // --- ML path: store results ---
  // Build lookup by grid_id
  const mlByGridId = Object.fromEntries(mlResults.map(r => [r.grid_id, r]));

  const insert = db.prepare(`
    INSERT INTO risk_scores(grid_id,timestamp,rainfall_score,soil_score,slope_score,historical_score,citizen_score,composite_score,risk_level,primary_factor,ml_probability,model_version)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
  `);

  const scores = [];
  for (const g of grids) {
    const ml = mlByGridId[g.id];
    if (!ml) continue;

    const fv = featureVectors.find(v => v.grid_id === g.id);

    // composite_score = probability × 100 for backward UI compat
    const composite = parseFloat((ml.probability * 100).toFixed(2));

    // Retain legacy factor scores for frontend charts (computed from sensor data)
    const sensor = db.prepare(
      'SELECT * FROM sensor_readings WHERE grid_id=? ORDER BY timestamp DESC LIMIT 1'
    ).get(g.id);
    const rain1h  = sensor?.rainfall_1h_mm  || 0;
    const rain24h = sensor?.rainfall_24h_mm || 0;
    const sm      = sensor?.soil_moisture   || 0.3;

    const rainfallScore  = parseFloat((Math.min(1.0, (rain1h / 50) * 0.4 + (rain24h / 150) * 0.6) * 100).toFixed(2));
    const soilScore      = parseFloat((sm * 100).toFixed(2));
    const slopeScore     = parseFloat((Math.min(1.0, g.slope_angle / 45) * 100).toFixed(2));
    const historicalScore= parseFloat((Math.min(1.0, g.historical_incidents / 12) * 100).toFixed(2));

    const recentReports = db.prepare(
      `SELECT COUNT(*) as cnt FROM citizen_reports WHERE grid_id=? AND submitted_at > datetime('now', '-24 hours')`
    ).get(g.id);
    const citizenScore = parseFloat((Math.min(1.0, (recentReports?.cnt || 0) / 3) * 100).toFixed(2));

    // Primary factor from ML-weighted perspective (still use old factor names for UI compat)
    const factors = [
      { name: 'Rainfall',             score: rainfallScore },
      { name: 'Soil Moisture',        score: soilScore },
      { name: 'Slope Angle',          score: slopeScore },
      { name: 'Historical Incidents', score: historicalScore },
      { name: 'Citizen Reports',      score: citizenScore },
    ];
    const primaryFactor = factors.sort((a, b) => b.score - a.score)[0].name;

    insert.run(
      g.id, now,
      rainfallScore, soilScore, slopeScore, historicalScore, citizenScore,
      composite, ml.risk_level, primaryFactor,
      ml.probability, ml.model_version || modelVersion
    );

    scores.push({
      grid_id:          g.id,
      timestamp:        now,
      rainfall_score:   rainfallScore,
      soil_score:       soilScore,
      slope_score:      slopeScore,
      historical_score: historicalScore,
      citizen_score:    citizenScore,
      composite_score:  composite,
      risk_level:       ml.risk_level,
      primary_factor:   primaryFactor,
      ml_probability:   ml.probability,
      model_version:    ml.model_version || modelVersion,
      degraded_mode:    false,
      lat:              g.lat,
      lng:              g.lng,
      district:         g.district,
      slope_angle:      g.slope_angle,
      elevation:        g.elevation,
      road_proximity_km:    g.road_proximity_km,
      village_proximity_km: g.village_proximity_km,
    });
  }

  return scores;
}

// ---------------------------------------------------------------------------
// Get latest risk scores for all grids (with location info)
// ---------------------------------------------------------------------------
export function getLatestRiskScores() {
  const db = getDB();
  return db.prepare(`
    SELECT rs.*, gc.lat, gc.lng, gc.district, gc.slope_angle, gc.elevation, gc.road_proximity_km, gc.village_proximity_km
    FROM risk_scores rs
    JOIN grid_cells gc ON rs.grid_id = gc.id
    WHERE rs.id IN (SELECT MAX(id) FROM risk_scores GROUP BY grid_id)
    ORDER BY rs.composite_score DESC
  `).all();
}
