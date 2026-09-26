import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

const EXPECTED_FEATURE_KEYS = [
  'elevation_m', 'slope_deg', 'aspect_deg', 'plan_curvature', 'profile_curvature',
  'twi', 'dist_to_stream_m', 'land_cover_code', 'soil_type_enc', 'rainfall_1h_mm',
  'rainfall_3h_mm', 'rainfall_6h_mm', 'rainfall_12h_mm', 'rainfall_24h_mm',
  'rainfall_3d_accum_mm', 'rainfall_7d_accum_mm', 'rainfall_intensity_mm_h',
  'soil_moisture_pct', 'historical_landslide_density', 'rain_moisture_index',
  'slope_wetness_index', 'rainfall_runoff_proxy',
];

test('risk scoring batches primary ML inference and falls back when the service is down', async () => {
  const tempDirectory = mkdtempSync(join(tmpdir(), 'dhara-risk-engine-'));
  process.env.DATABASE_PATH = join(tempDirectory, 'integration.sqlite');

  const requestBatches = [];
  const mockService = createServer((req, res) => {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      if (req.method !== 'POST' || req.url !== '/predict') {
        res.writeHead(404).end();
        return;
      }
      const records = JSON.parse(body).records;
      requestBatches.push(records);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        predictions: records.map(record => ({
          grid_id: record.grid_id,
          probability: 0.91,
          risk_level: 'Critical',
          model_version: 'mock-ml-service',
          threshold: 0.46,
          features_used: record,
        })),
      }));
    });
  });
  await new Promise(resolve => mockService.listen(0, '127.0.0.1', resolve));
  process.env.ML_SERVICE_URL = `http://127.0.0.1:${mockService.address().port}`;

  let database;
  const logMessages = [];
  const originalInfo = console.info;
  const originalWarn = console.warn;
  console.info = (...args) => logMessages.push(args.join(' '));
  console.warn = (...args) => logMessages.push(args.join(' '));

  try {
    const { initDB, getDB } = await import('../db/database.js');
    const { computeRiskScores, getLatestRiskScores, isRiskEngineFallbackActive, simulateSensorFeed } =
      await import('../services/riskEngine.js');
    database = initDB();

    const readings = simulateSensorFeed();
    const primaryScores = await computeRiskScores();
    assert.ok(readings.length > 0);
    assert.equal(requestBatches.length, 1, 'all grids should be sent in one ML request');
    assert.equal(requestBatches[0].length, readings.length);
    assert.deepEqual(Object.keys(requestBatches[0][0]).filter(key => key !== 'grid_id').sort(), [...EXPECTED_FEATURE_KEYS].sort());
    assert.ok(logMessages.some(message => message.includes('[RiskEngine] Using ml-service')));
    assert.equal(isRiskEngineFallbackActive(), false);
    assert.equal(primaryScores.length, readings.length);
    assert.ok(primaryScores.every(score => score.ml_probability === 0.91));
    assert.equal(getLatestRiskScores().length, readings.length, 'primary predictions should be persisted and queryable');

    await new Promise((resolve, reject) => mockService.close(error => error ? reject(error) : resolve()));
    const fallbackScores = await computeRiskScores();
    assert.ok(logMessages.some(message => message.includes('[RiskEngine] ml-service unavailable, using local XGBoost fallback')));
    assert.equal(isRiskEngineFallbackActive(), true);
    assert.equal(fallbackScores.length, readings.length);
    assert.ok(fallbackScores.every(score => Number.isFinite(score.ml_probability)));
    assert.equal(getLatestRiskScores().length, readings.length, 'fallback predictions should also be persisted');

    assert.ok(getDB());
  } finally {
    console.info = originalInfo;
    console.warn = originalWarn;
    if (mockService.listening) {
      await new Promise(resolve => mockService.close(resolve));
    }
    database?.close();
    rmSync(tempDirectory, { recursive: true, force: true });
    delete process.env.DATABASE_PATH;
    delete process.env.ML_SERVICE_URL;
  }
});
