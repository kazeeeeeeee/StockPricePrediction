# Time Series Modelling to Stock Price

In this project, my aim is to develop a reference model that investors can utilize to make informed decisions regarding the potential purchase of a specific stock.


## Background

In Singapore, the aspiration for achieving early retirement is steadily gaining traction. Investment avenues like the property market and stock market have become popular choices for enhancing supplementary income streams.

However, the property market demands substantial capital investment and often entails extended turnaround times for resale, rendering it less suitable for individuals with limited cashflow. Consequently, the stock market presents a more accessible and early-starting alternative.

Amidst the plethora of stock reviews and recommendations offered by esteemed advisors, a diverse array of perspectives and subjective factors contribute to varying endorsements. For example, various interpretations of key financial parameters, such as the Price-to-Earnings (P/E) ratio, lack universal consensus on what constitutes high or low values.

In light of these considerations, I am motivated to develop my own model for Stock Market Prediction.


## Problem Statement

**Objective**
Formulate an exemplary model that empowers investors to arrive at well-informed decisions concerning the potential buy, sell, or hold strategies for a specific stock.

Initiating the model's scope, the initial focus will be solely on the NASDAQ market.
The recommended decisions will fall into these three categories:

***Purchase (Phase 1)***
- This recommendation stands out as relatively straightforward, as it suggests buying in instances where there is a projected increase in the stock's price.

***Sell/Hold (Phase 2)***
- This phase involves incorporating the user's (investor's) input of the current Weighted Average Cost of the particular stock. This information will be compared against the present stock price and its predicted price. Furthermore, factors like the possibility of selling the stock for alternative investment opportunities will also come into play.
Conceptual Framework: Time Series Analysis

**Performance Metrics:**
- Mean Absolute Percentage Error (MAPE), Root Mean Square Error (RMSE), and Bias for price prediction evaluation.
- F1-score will be employed to assess the effectiveness of buy recommendations.


## Data Collection

Downloads from:
- https://eoddata.com/download.aspx
    > - downloads the entire list of NASDAQ tickers in CSV format

Webscrape using API from 2 sources:
- https://rapidapi.com/twelvedata/api/twelve-data1
    > - To get weekly 'open', 'high', 'low', 'close', 'date' and 'volume' traded
        >> - url: "https://twelve-data1.p.rapidapi.com/time_series"
    > - To get 'name', 'sector', 'industry', 'description', 'type'
        >> - url: "https://twelve-data1.p.rapidapi.com/profile"
    > - To get dividend 'amount' and 'payment date'
        >> - url: "https://twelve-data1.p.rapidapi.com/dividends"
- yfinance api
    > - To get income statments
        >> - yf.Ticker("symbol").income_stmt
    > - To get balance sheets
        >> - yf.Ticker("symbol").balance_sheet
    > - To get cash flow
        >> - yf.Ticker("symbol").cashflow

Assumption made: data sources are accurate.


## Data Cleaning

### Stock profile dataset
This dataset consists of stock symbol, name, sector, industry, description, type, CEO and country. These data have missing values for name, sector, industry, description and CEO. Since these are not used in the modelling process, the data with missing values were retained.

### Stock dividend dataset
This dataset consists of the stock symbol, dividend payment date and amount. There are missing symbols from the API webscraping. Upon investigating the raw CSV file from the webscraping, the missing symbols are unable to be determined if they belong to the Nano Labs Ltd which has the symbol "NA". Hence, these missing data are removed.

### Stock historical price dataset
This dataset consists of the date, stock symbol, open price, high price, low price, close price, volume, and currency. There are missing symbol and volume data from the API webscraping.

Upon investigating the raw CSV file from the webscraping, the missing symbols are unable to be determined if they belong to the Nano Labs Ltd which has the symbol "NA". Hence, these missing data are removed. The missing volume data were kept intact since this was not part of the model parameters.

There were 1048 symbols with duplicated data on the same date. E.g. ZYME has 2 rows of data for 2017-04-24. The data is cleaned to only keep the 1st row of data for such occurences for all the 1048 symbols.

### Stock income statement dataset
This dataset consists of a lot of data such as date, stock symbol, tax effect of unusual items, tax rate for calcs, normalized EBITDA, total unusal items, total unusual items excluding goodwill, net income from continuing operation net minority interest, reconciled depreciation, reconciled cost of revenue, EBIT, net interest income, normalized income, net income from continuing and discontinued operation, and etc. There are 3 missing symbols and a lot of missing data in other features from the API webscraping.

Upon investigating the raw CSV file from the webscraping, the missing symbols actually belong to the Nano Labs Ltd which has the symbol "NA" which were interpreted as "NaN" when imported into the Jupyter Notebook. The other missing data were kept intact since these are not part of the model parameters and it is expected that not all companies would have all the fields e.g. "gain on sale of business" which refers to the profit realized when a business owner sells their business for an amount that exceeds the original purchase cost (basis) of the business.

### Stock balance sheet dataset
This dataset consists of a lot of data such as date, stock symbol, treasury shares number, ordinary shares number, tangible book value, working capital, and etc. There are 3 missing symbols and a lot of missing data in other features from the API webscraping.

Upon investigating the raw CSV file from the webscraping, the missing symbols actually belong to the Nano Labs Ltd which has the symbol "NA" which were interpreted as "NaN" when imported into the Jupyter Notebook. The other missing data were kept intact since these are not part of the model parameters and it is expected that not all companies would have all the fields e.g. "fixed assets revaluation reserve" which record changes in the value of a company's fixed assets due to revaluation.

### Stock cashflow dataset
This dataset consists of a lot of data such as date, free cash flow, issuance of debt, capital expenditure, and etc. There are 3 missing symbols and a lot of missing data in other features from the API webscraping.

Upon investigating the raw CSV file from the webscraping, the missing symbols actually belong to the Nano Labs Ltd which has the symbol "NA" which were interpreted as "NaN" when imported into the Jupyter Notebook. The other missing data were kept intact since these are not part of the model parameters and it is expected that not all companies would have all the fields e.g. "cash from discontinued investing activities" refers to the cash flows generated or used in investment activities that are classified as discontinued operations.


## EDA
There are 12 sectors and 137 industry groupings. Of which, most of the sectors belong to healthcare, financial services and technology, adding up to more than 60% of the total symbols on NASDAQ market. In terms of industry grouping, Biotechnology has the most symbols (18%) follow by Banks-Regional (8%), Software-Application (5%). Most of the symbols are US companies (99.83%).

Most dividend payout are less than $1 per stock. This suggest that the amount of dividend is unlikely to affect the popularity and close price of the stocks.

There are 68 stocks consistently below $1 for all month-end closing and 214 stocks consistently below $1 for all year-end closing for the past 10 years. Stocks that are typically lesser than $1 are considered as penny stocks. There is no observable trends across all these penny stocks but it is noted that most of these stocks only start listing recently. This means that usually stocks do not remain as penny stock in its entire lifetime.

There are 461 stocks consistently below $5 for all year-end closing and 829 stocks consistently below $10 for all year-end closing for the past 10 years. This list has quite a a lot of symbols consistently "low valued" since 2014. This means some stocks may never "climax" its entire lifetime.

Finally, there are 14 stocks consistently more than $100 for all month-end closing and 32 stocks consistently above $100 for all year-end closing for the past 10 years. There is no observable trends across all these "high valued" stocks but there is generally greater price fluctuations across the years as compared to the penny stocks. These stocks are subjected to high loss or high profit in short term investment.

There are alot of missing data from various datasets mentioned above and it would be difficult to use these to develop a model suited for all stocks.


## Model

A total of 8 models were run on TSLA stock to select the best model to replicate the prediction of stock closing price for the rest of the NASDAQ stocks.

The parameters used were closing/opening/high/low price of 6 months to 12 months ago. Data from 2013-01-28 to 2020-08-31 is used for training while data from 2020-09-30 to 2023-07-24 is used for testing.

Linear Regression Model was first established as the base model. This yields MAPE=1.82, RMSE=495.81, Bias=-429.38. This model's close price predictions are deviating too much from the actual close price. Another 7 models were explored: ARMIA, ARIMAX, SARIMAX, Simple Exponential Smoothing, Prophet Linear Model, Prophet Logistic Model, and VARIMA. MAPE, RMSE and Bias are metrics used to evaluate the close price predictions.

VARIMA from the Darts library was the best model, yielding MAPE=0.22, RMSE=62.71, Bias=2.92. Grid search was then applied to finetune the hyperparameters of p (order of the autoregressive component), d (order of differencing), q (order of the moving average component) and num_samples (number of times a prediction is sampled from a probabilistic model). However, VARIMA with the original hyperparameters of p=1, d=1, q=0, num_samples=1 still produce the best result. This model is then used for all other stocks.

Out of the total of 4330 stocks from NASDAQ, 2220 stocks managed to fit the VARIMA model successfully while the other 2110 stocks have missing dates and hence return errors when fitting the model. Due to time constraint, analysis is done only on the 2220 stocks that fit the VARIMA model successfully.

The predicted price for the stocks are then compared against the stock price as of 2020-08-31. If the predicted price is higher, it will be a buy recommendation. This is then checked against the actual close price. Should the actual close price be also higher than the stock price as of 2020-08-31, it will mean that the buy recommendation is accurate. Using the confusion matrix, F1-score is used to determine the accuracy of the recommendation.


## Results

After completion of the forecast model, 478 stocks with good F1_score (greater than or equal to 0.9) are identified. With this list of stocks, the VARIMA model (p=1, d=1, q=0, num_samples=1) can be used to ingest approximately past 6 years of months' month-end stock price (OHLC) to predict the close price 6 months thereafter and beyond, then compare this prediction against the current stock price to recommend investors should they buy the particular stock.


## Conclusion

The model established is quite simple and hence the prediction would not be able to factor in a lot other factors. In future, I will explore more complicated models, ingest more parameters such as the 'sector', 'industry' and age of the stock/company, and to develop more robust model to account for missing information such as dates. It is also noted that I would need better understanding of all the various information e.g. finance terms to better appreciate the parameters and consideration for modelling.

Nevertheless, the approach of identifying stocks with higher accuracy for buy recommendation is a good starting point for novice investors to refer amongst other stock review sources to make informed decision.

However, it is important to note that stock price is influenced by many factors including macroeconomic, geopolitical and investors' sentiments which can be very unpredictable. Therefore stock investment's inherent risk is always present.
