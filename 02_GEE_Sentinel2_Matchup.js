```javascript
// ======================================
// In-situ data extraction (CSV time string filtering)
// ======================================
var pts = ee.FeatureCollection("projects/wt-rs-inv-2025/assets/gee_matched_turbidity")
  .filter(ee.Filter.stringStartsWith('Date', '2021-03')); // Filter by target month

var win = 3;        // ±days for time window matching
var minPix = 4;     // Minimum valid pixels for reliable sampling
var bufR = 400;     // Buffer radius for point sampling (meters)

// ======================================
// Permanent water mask initialization
// ======================================
var water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
  .select('occurrence')
  .gt(5); // Water occurrence > 5%

// ======================================
// Sentinel-2 image preprocessing function
// ======================================
function prepS2(img){
  // Reflectance calibration and band renaming
  var refl = img.select(['B2','B3','B4','B8','B11','B12'])
    .multiply(0.0001)
    .rename(['Blue','Green','Red','NIR','SWIR1','SWIR2']);

  var scl = img.select('SCL'); // Scene Classification Layer

  // Cloud/shadow/noise masking
  var mask = scl.neq(8)  // Cloud medium prob
    .and(scl.neq(9))     // Cloud high prob
    .and(scl.neq(10))    // Cirrus
    .and(scl.neq(11))    // Snow/Ice
    .and(scl.neq(1))     // Saturated/Defective
    .and(scl.neq(3));    // Cloud shadows

  // Apply masks and set metadata
  return refl
    .updateMask(mask)
    .updateMask(water)
    .set({
      'Satellite':'Sentinel2',
      'system:time_start': img.get('system:time_start')
    });
}

// ======================================
// Sentinel-2 Surface Reflectance collection
// ======================================
var s2Col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");

// ======================================
// Core in-situ and satellite matching function
// ======================================
function extract(points, col, scale, prepFn){
  return points.map(function(pt){
    // Parse in-situ timestamp
    var cleanDateStr = ee.String(pt.get('Date')).trim();
    var d = ee.Date.parse('yyyy-MM-dd HH:mm:ss', cleanDateStr);

    var buf = pt.geometry().buffer(bufR); // Create sampling buffer

    // Filter and preprocess satellite images
    var imgs = col
      .filterBounds(buf)
      .filterDate(d.advance(-win,'day'), d.advance(win,'day'))
      .map(prepFn);

    // Extract statistical metrics per image
    return imgs.map(function(img){
      var stat = img.reduceRegion({
        reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true)
          .combine(ee.Reducer.count(), '', true),
        geometry: buf,
        scale: scale,
        maxPixels: 1e6
      });

      // Merge point properties and satellite metrics
      return ee.Feature(pt.geometry(), pt.toDictionary())
        .set(stat)
        .set('Satellite_Date', img.date().format('yyyy-MM-dd HH:mm:ss'));
    });
  }).flatten(); // Flatten nested feature collections
}

// ======================================
// Execute Sentinel-2 matching and quality control
// ======================================
var matchS2 = extract(pts, s2Col, 20, prepS2)
  .filter(ee.Filter.notNull(['Red_mean']))  // Remove null values
  .filter(ee.Filter.gte('Red_count', minPix)); // Filter by minimum pixels

// ======================================
// Export matched results to Google Drive
// ======================================
Export.table.toDrive({
  collection: matchS2,
  description: 'S_2021_03',
  folder: 'GEE_Outputs',
  fileFormat: 'CSV'
});
```