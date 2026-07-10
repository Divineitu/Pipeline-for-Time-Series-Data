// Query 1: Fetch Latest Record sorted by Date cadence
db.sales_records.find({}).sort({ date_friday: -1 }).limit(1);

// Query 2: Fetch Records by strict Date Range boundary
db.sales_records.find({
  date_friday: { $gte: "2012-10-01", $lte: "2012-10-26" },
});

// Query 3: Find Holiday documents matching heavy promotional markdown conditions
db.sales_records.find({
  "environmental_features.is_holiday": true,
  "sales_data.actual_weekly_sales": { $gt: 100000.0 },
});
