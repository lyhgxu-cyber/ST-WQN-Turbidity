```javascript
// ======================================
// In-situ Data Matching & Spectral Extraction
// Landsat 8/9 Time-Series Processing
// ======================================

var pts = ee.FeatureCollection("projects/wt-rs-inv-2025/assets/gee_matched_turbidity")
  .filter(ee.Filter.stringStartsWith('Date', '2021'));

var win = 3;             // Temporal window ± days
var minPix = 4;          // Minimum valid pixel threshold
var bufR = 400;          // Spatial buffer radius (meters)

// JRC Global Surface Water Mask
var water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
  .select('occurrence')
  .gt(5);

// ======================================
// Landsat 8/9 Preprocessing Pipeline
// ======================================
function prepL89(img){
  // Surface reflectance calibration & band renaming
  var refl = img.select([
      'SR_B2','SR_B3','SR_B4',
      'SR_B5','SR_B6','SR_B7'
    ])
    .multiply(0.0000275)
    .add(-0.2)
    .rename([
      'Blue','Green','Red',
      'NIR','SWIR1','SWIR2'
    ]);

  // QA Pixel cloud & shadow masking
  var qa = img.select('QA_PIXEL');
  var cloudMask = qa.bitwiseAnd(1<<4).eq(0)
    .and(qa.bitwiseAnd(1<<3).eq(0));

  // Apply masks and metadata
  return refl
    .updateMask(cloudMask)
    .updateMask(water)
    .set({
      'Satellite':'Landsat',
      'system:time_start': img.get('system:time_start')
    });
}

// Merged Landsat 8/9 Collection
var l89Col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"));

// ======================================
// Core Spatial-Temporal Extraction Function
// ======================================
function extract(points){
  return points.map(function(pt){
    // Parse in-situ timestamp
    var d = ee.Date.parse('yyyy-MM-dd HH:mm:ss',
      ee.String(pt.get('Date')).trim());
    
    var buf = pt.geometry().buffer(bufR);

    // Temporal-spatial image filtering
    var imgs = l89Col
      .filterBounds(buf)
      .filterDate(d.advance(-win,'day'), d.advance(win,'day'))
      .map(prepL89);

    // Pixel statistics extraction
    return imgs.map(function(img){
      var stat = img.reduceRegion({
        reducer: ee.Reducer.mean()
          .combine(ee.Reducer.stdDev(),'',true)
          .combine(ee.Reducer.count(),'',true),
        geometry: buf,
        scale: 30,
        maxPixels: 1e6
      });

      // Assemble output feature
      return ee.Feature(pt.geometry(), pt.toDictionary())
        .set(stat)
        .set({
          'Image_Time': img.get('system:time_start'),
          'Satellite_Date': img.date().format('yyyy-MM-dd HH:mm:ss')
        });
    });
  }).flatten();
}

// ======================================
// Quality Control & Result Filtering
// ======================================
var matchL89 = extract(pts)
  .filter(ee.Filter.notNull(['Red_mean']))
  .filter(ee.Filter.gte('Red_count', minPix));

// ======================================
// Export Final Dataset to Drive
// ======================================
Export.table.toDrive({
  collection: matchL89,
  description: '2021_L89',
  folder: 'GEE_Outputs1',
  fileFormat: 'CSV'
});
```