// ======================================
// In-situ 数据读取（CSV时间字符串过滤）
// ======================================
var pts = ee.FeatureCollection("projects/wt-rs-inv-2025/assets/gee_matched_turbidity")
  .filter(ee.Filter.stringStartsWith('Date', '2021-03'));

var win = 3;        // ±days
var minPix = 4;     // 最少有效像元
var bufR = 400;     // buffer半径（m）

// ======================================
// 水体掩膜
// ======================================
var water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
  .select('occurrence')
  .gt(5);

// ======================================
// Sentinel-2 预处理函数
// ======================================
function prepS2(img){

  var refl = img.select(['B2','B3','B4','B8','B11','B12'])
    .multiply(0.0001)
    .rename(['Blue','Green','Red','NIR','SWIR1','SWIR2']);

  var scl = img.select('SCL');

  var mask = scl.neq(8)
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11))
    .and(scl.neq(1))
    .and(scl.neq(3));

  return refl
    .updateMask(mask)
    .updateMask(water)
    .set({
      'Satellite':'Sentinel2',
      'system:time_start': img.get('system:time_start')
    });
}

// ======================================
// Sentinel-2 影像集合
// ======================================
var s2Col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");

// ======================================
// 核心匹配函数
// ======================================
function extract(points, col, scale, prepFn){

  return points.map(function(pt){

    var cleanDateStr = ee.String(pt.get('Date')).trim();
    var d = ee.Date.parse('yyyy-MM-dd HH:mm:ss', cleanDateStr);

    var buf = pt.geometry().buffer(bufR);

    var imgs = col
      .filterBounds(buf)
      .filterDate(
        d.advance(-win,'day'),
        d.advance(win,'day')
      )
      .map(prepFn);

    return imgs.map(function(img){

      var stat = img.reduceRegion({
        reducer: ee.Reducer.mean()
          .combine(ee.Reducer.stdDev(), '', true)
          .combine(ee.Reducer.count(), '', true),
        geometry: buf,
        scale: scale,
        maxPixels: 1e6
      });

      return ee.Feature(pt.geometry(), pt.toDictionary())
        .set(stat)
        .set('Satellite_Date', img.date().format('yyyy-MM-dd HH:mm:ss'));

    });

  }).flatten();
}

// ======================================
// Sentinel-2 提取
// ======================================
var matchS2 = extract(pts, s2Col, 20, prepS2)
  .filter(ee.Filter.notNull(['Red_mean']))
  .filter(ee.Filter.gte('Red_count', minPix));

// ======================================
// 导出 Sentinel-2
// ======================================
Export.table.toDrive({
  collection: matchS2,
  description: 'S_2021_03',
  folder: 'GEE_Outputs',
  fileFormat: 'CSV'
});