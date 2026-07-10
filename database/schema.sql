-- Create the Relational Database for Walmart Store Sales
CREATE DATABASE IF NOT EXISTS walmart_sales_db;
USE walmart_sales_db;

-- 1. Store Metadata Table (Normalization Entity)
CREATE TABLE IF NOT EXISTS stores (
    store_id INT PRIMARY KEY,
    store_type CHAR(1) NOT NULL,
    size_sq_ft INT NOT NULL
);

-- 2. Weekly External Macro & Environmental Features Table
CREATE TABLE IF NOT EXISTS weekly_features (
    feature_id INT AUTO_INCREMENT PRIMARY KEY,
    store_id INT NOT NULL,
    date_friday DATE NOT NULL,
    temperature DECIMAL(5,2),
    fuel_price DECIMAL(5,3),
    cpi DECIMAL(8,5),
    unemployment DECIMAL(5,3),
    is_holiday TINYINT(1) DEFAULT 0,
    markdown_1 DECIMAL(10,2) DEFAULT 0.00,
    markdown_2 DECIMAL(10,2) DEFAULT 0.00,
    markdown_3 DECIMAL(10,2) DEFAULT 0.00,
    markdown_4 DECIMAL(10,2) DEFAULT 0.00,
    markdown_5 DECIMAL(10,2) DEFAULT 0.00,
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    UNIQUE KEY unique_store_week (store_id, date_friday)
);

-- 3. Historical Department Sales & Forecast Target Table
CREATE TABLE IF NOT EXISTS department_sales (
    sales_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    store_id INT NOT NULL,
    dept_id INT NOT NULL,
    date_friday DATE NOT NULL,
    weekly_sales DECIMAL(12,2) NOT NULL,
    lag_1_sales DECIMAL(12,2) DEFAULT 0.00,
    rolling_4wk_mean DECIMAL(12,2) DEFAULT 0.00,
    predicted_sales DECIMAL(12,2) DEFAULT NULL,
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (store_id, date_friday) REFERENCES weekly_features(store_id, date_friday)
);