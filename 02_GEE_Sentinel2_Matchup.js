var pts = ee.FeatureCollection("projects/wt-rs-inv-2025/assets/gee_matched_turbidity")
  .filter(ee.Filter.stringStartsWith('Date', '2021'));

var win = 3;
var minPix = 4;
var bufR = 400;

var water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
  .select('occurrence')
  .gt(5);

// Landsat 预处理
function prepL89(img){

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

  var qa = img.select('QA_PIXEL');

  var cloudMask = qa.bitwiseAnd(1<<4).eq(0)
    .and(qa.bitwiseAnd(1<<3).eq(0));

  return refl
    .updateMask(cloudMask)
    .updateMask(water)
    .set({
      'Satellite':'Landsat',
      'system:time_start': img.get('system:time_start')
    });
}

var l89Col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"));

function extract(points){

  return points.map(function(pt){

    var d = ee.Date.parse('yyyy-MM-dd HH:mm:ss',
      ee.String(pt.get('Date')).trim());

    var buf = pt.geometry().buffer(bufR);

    var imgs = l89Col
      .filterBounds(buf)
      .filterDate(d.advance(-win,'day'), d.advance(win,'day'))
      .map(prepL89);

    return imgs.map(function(img){

      var stat = img.reduceRegion({
        reducer: ee.Reducer.mean()
          .combine(ee.Reducer.stdDev(),'',true)
          .combine(ee.Reducer.count(),'',true),
        geometry: buf,
        scale: 30,
        maxPixels: 1e6
      });

      return ee.Feature(pt.geometry(), pt.toDictionary())
        .set(stat)
        .set({
          'Image_Time': img.get('system:time_start'),
          'Satellite_Date': img.date().format('yyyy-MM-dd HH:mm:ss')
        });

    });

  }).flatten();

}

var matchL89 = extract(pts)
  .filter(ee.Filter.notNull(['Red_mean']))
  .filter(ee.Filter.gte('Red_count', minPix));

Export.table.toDrive({
  collection: matchL89,
  description: '2021_L89',
  folder: 'GEE_Outputs1',
  fileFormat: 'CSV'
});