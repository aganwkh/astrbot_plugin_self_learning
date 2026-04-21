const assert = require("assert");
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const appPath = path.resolve(
  __dirname,
  "..",
  "..",
  "web_res",
  "static",
  "js",
  "macos",
  "apps",
  "StyleLearning.js",
);

const code = fs.readFileSync(appPath, "utf8");
const sandbox = { window: {}, console };
vm.createContext(sandbox);
vm.runInContext(code, sandbox);

const utils = sandbox.window.StyleLearningDataUtils;

assert.ok(utils, "StyleLearningDataUtils should be exposed on window");

const normalizedResults = utils.normalizeResultsPayload({
  statistics: {
    style_type_count: 3,
    average_confidence: null,
    raw_message_count: 0,
    last_updated_at: null,
    confidence_value_count: 0,
  },
  progress: [],
});
assert.strictEqual(normalizedResults.style_types_count, 3);
assert.strictEqual(normalizedResults.avg_confidence, null);
assert.strictEqual(normalizedResults.total_samples, 0);
assert.strictEqual(normalizedResults.raw_message_count_source, "unavailable");

const normalizedProgress = utils.normalizeProgressItems([
  {
    label: "batch_a",
    quality_score: 0.8,
    filtered_count: 12,
  },
  {
    label: "batch_b",
    filtered_count: 9,
  },
]);
assert.strictEqual(normalizedProgress[0].quality_percent, 80);
assert.strictEqual(normalizedProgress[0].sample_count, 12);
assert.strictEqual(normalizedProgress[1].quality_percent, null);
assert.strictEqual(normalizedProgress[1].quality_value_source, "missing");
const zeroQuality = utils.getProgressQualityValue({ quality_score: 0 });
assert.strictEqual(zeroQuality.score, 0);
assert.strictEqual(zeroQuality.percent, 0);
assert.strictEqual(zeroQuality.source, "quality_score");
const historicalMissingQuality = utils.getProgressQualityValue({
  label: "legacy",
  quality_value_source: "historical_missing",
});
assert.strictEqual(historicalMissingQuality.score, null);
assert.strictEqual(historicalMissingQuality.source, "historical_missing");

const normalizedPatterns = utils.normalizePatternsPayload({
  emotion_patterns: [],
  language_patterns: [],
  topic_preferences: [],
});
assert.deepStrictEqual(normalizedPatterns.topic_patterns, []);
assert.strictEqual(utils.getPatternNumericValue({ name: "fake" }), null);
assert.strictEqual(utils.getPatternNumericValue({ name: "real", count: "3" }), 3);
assert.strictEqual(
  utils.getPatternLabel('{"display_name":"clean label"}'),
  "clean label",
);
assert.strictEqual(
  utils.getPatternPercent({ confidence: 0 }),
  0,
);
const missingProgressQuality = utils.getProgressQualityValue({ label: "missing" });
assert.strictEqual(missingProgressQuality.score, null);
assert.strictEqual(missingProgressQuality.percent, null);
assert.strictEqual(missingProgressQuality.source, "missing");
