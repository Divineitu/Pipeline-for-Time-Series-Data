-- Query 1: Fetch Latest Time Series Record (Latest Friday Sales)
SELECT * FROM department_sales 
ORDER BY date_friday DESC LIMIT 1;

-- Query 2: Fetch Records by Date Range
SELECT * FROM department_sales 
WHERE date_friday BETWEEN '2012-10-01' AND '2012-10-26';

-- Query 3: Complex Aggregation for High-Performing Holiday Sales
SELECT s.store_id, d.dept_id, d.weekly_sales, f.unemployment
FROM department_sales d
JOIN weekly_features f ON d.store_id = f.store_id AND d.date_friday = f.date_friday
JOIN stores s ON d.store_id = s.store_id
WHERE f.is_holiday = 1 AND d.weekly_sales > 100000.00;