# New York Real Estate Data Analysis and Visualization Requirements Document

### Overview

- `index.html`: Data Analysis Dashboard

- `predict.html`: Price Prediction

## 1 index.html

### 1.1 Overall Page Requirements

- The page must display **7 charts**, and all must render correctly (no empty charts allowed).

- The page needs to provide an entry button/link to jump to the prediction page: `/predict`

- The page contains 3 tabs for switching between displaying 3 charts (after clicking, the charts display correctly and adapt to the container size).

- **Chart types cannot be duplicated**: The same chart type (e.g., line/scatter/bar, etc.) can appear at most once on the page.

### 1.2 Visualization Chart Requirements

1. **Average Sale Price by District (Bar Chart)**

- Purpose: To compare the average sale price levels of Manhattan/Brooklyn/Bronx/Queens/Staten Island

- Data: Aggregated by BOROUGH, calculate SALE PRICE 1. Average

2. **Building Category Percentage (Pie Chart)**

- Purpose: To view the most common building class categories in transactions.

- Data: Count the number of building class categories; display the top N, and merge the rest into Other (to avoid too many categories).

3. **Area and Transaction Price Relationship (Scatter Plot)**

- Purpose: To observe whether there is a correlation between area (gross square feet) and transaction price (sale price), and whether there are obvious outliers.

- Data: Sample from cleaned data (e.g., a maximum of 2000 records) to avoid too dense a point; can be colored by BOROUGH_NAME.

4. **Monthly Average Price Trend (Line Chart)**

- Purpose: To observe the trend and fluctuation of transaction prices over time.

- Data: Convert sale dates to datetime; aggregate and calculate the average sale price by month.

5. **Transaction Price Distribution (Histogram, log10)**

- Purpose: To display the overall transaction price distribution pattern (log10 processing makes long-tail data easier to observe)

- Data: Histogram plotted after taking log10 of SALE PRICE (sample size can be limited to avoid lag)

6. **Unit Price Distribution by Administrative Region (Box Plot)**

- Purpose: To compare the distribution of "unit price ($/SqFt)" in different administrative regions and visually display outliers

- Data: Price_Per_SqFt = SALE PRICE / GROSS SQUARE FEET; box plots plotted by grouping by BOROUGH_NAME

7. **Feature Correlation (Heatmap)**

- Purpose: To quickly determine the linear correlation (positive/negative/weak correlation) between key features

- Data: SALE PRICE, GROSS SQUARE FEET, YEAR BUILT, TOTAL UNITS, Price_Per_SqFt Calculate the correlation coefficient matrix (corr)

### 1.3 Data Cleaning and Performance Requirements (Brief)

- Data cleaning should include at least: deduplication, converting numerical columns to numeric values, and filtering obviously invalid values (such as SALE PRICE <= 10000, area <= 0, YEAR BUILT <= 1800, etc.).

- The amount of visualization data needs to be controlled: scatter plots must be sampled; other charts should aggregate or limit samples to avoid front-end crashes.

## 2 predict.html

### 2.1 Page Functionality

- After the user inputs features and clicks "Predict Price," the page displays the predicted **SALE PRICE** (thousands place recommended, retain 2 decimal places).

- Display a user-friendly error message when input is invalid (back-end crashes are not allowed).

### 2.2 Input Items and Validation Rules

- GROSS SQUARE FEET: Must > 0

- YEAR BUILT: Range 1800–2030

- TOTAL UNITS: Range 1–1000

- BOROUGH: Drop-down selection (1–5 or corresponding name)