import csv

# List of Binance API endpoints and their descriptions, including parameters and response structure
binance_api_endpoints = [
    {
        "endpoint": "GET /api/v3/depth",
        "description": "Order book",
        "parameters": [
            {"name": "symbol", "type": "STRING", "mandatory": "YES", "description": ""},
            {"name": "limit", "type": "INT", "mandatory": "NO", "description": "Default 100; max 5000. If limit > 5000. then the response will truncate to 5000."}
        ],
        "response": {
            "lastUpdateId": "1027024",
            "bids": [["4.00000000", "431.00000000"]],
            "asks": [["4.00000200", "12.00000000"]]
        }
    },
    {
        "endpoint": "GET /api/v3/trades",
        "description": "Recent trades list",
        "parameters": [
            {"name": "symbol", "type": "STRING", "mandatory": "YES", "description": ""},
            {"name": "limit", "type": "INT", "mandatory": "NO", "description": "Default 500; max 1000."}
        ],
        "response": [
            {
                "id": 28457,
                "price": "4.00000100",
                "qty": "12.00000000",
                "quoteQty": "48.000012",
                "time": 1499865549590,
                "isBuyerMaker": True,
                "isBestMatch": True
            }
        ]
    },
    {
        "endpoint": "GET /api/v3/historicalTrades",
        "description": "Old trade lookup",
        "parameters": [
            {"name": "symbol", "type": "STRING", "mandatory": "YES", "description": ""},
            {"name": "limit", "type": "INT", "mandatory": "NO", "description": "Default 500; max 1000."},
            {"name": "fromId", "type": "LONG", "mandatory": "NO", "description": "TradeId to fetch from. Default gets most recent trades."}
        ],
        "response": [
            {
                "id": 28457,
                "price": "4.00000100",
                "qty": "12.00000000",
                "quoteQty": "48.000012",
                "time": 1499865549590,
                "isBuyerMaker": True,
                "isBestMatch": True
            }
        ]
    },
    {
        "endpoint": "GET /api/v3/aggTrades",
        "description": "Compressed/Aggregate trades list",
        "parameters": [
            {"name": "symbol", "type": "STRING", "mandatory": "YES", "description": ""},
            {"name": "fromId", "type": "LONG", "mandatory": "NO", "description": "ID to get aggregate trades from INCLUSIVE."},
            {"name": "startTime", "type": "LONG", "mandatory": "NO", "description": "Timestamp in ms to get aggregate trades from INCLUSIVE."},
            {"name": "endTime", "type": "LONG", "mandatory": "NO", "description": "Timestamp in ms to get aggregate trades until INCLUSIVE."},
            {"name": "limit", "type": "INT", "mandatory": "NO", "description": "Default 500; max 1000."}
        ],
        "response": [
            {
                "a": 26129,
                "p": "0.01633102",
                "q": "4.70443515",
                "f": 27781,
                "l": 27781,
                "T": 1498793709153,
                "m": True,
                "M": True
            }
        ]
    },
    {
        "endpoint": "GET /api/v3/klines",
        "description": "Kline/Candlestick data",
        "parameters": [
            {"name": "symbol", "type": "STRING", "mandatory": "YES", "description": ""},
            {"name": "interval", "type": "ENUM", "mandatory": "YES", "description": ""},
            {"name": "startTime", "type": "LONG", "mandatory": "NO", "description": ""},
            {"name": "endTime", "type": "LONG", "mandatory": "NO", "description": ""},
            {"name": "limit", "type": "INT", "mandatory": "NO", "description": "Default 500; max 1000."}
        ],
        "response": [
            [
                1499040000000,
                "0.01634790",
                "0.80000000",
                "0.01575800",
                "0.01577100",
                "148976.11427815",
                1499644799999,
                "2434.19055334",
                308,
                "1756.87402397",
                "28.46694368",
                "0"
            ]
        ]
    },
    {
        "endpoint": "GET /api/v3/uiKlines",
        "description": "UIKlines",
        "parameters": [
            {"name": "symbol", "type": "STRING", "mandatory": "YES", "description": ""},
            {"name": "interval", "type": "ENUM", "mandatory": "YES", "description": ""},
            {"name": "startTime", "type": "LONG", "mandatory": "NO", "description": ""},
            {"name": "endTime", "type": "LONG", "mandatory": "NO", "description": ""},
            {"name": "limit", "type": "INT", "mandatory": "NO", "description": "Default 500; max 1000."}
        ],
        "response": [
            [
                1499040000000,
                "0.01634790",
                "0.80000000",
                "0.01575800",
                "0.01577100",
                "148976.11427815",
                1499644799999,
                "2434.19055334",
                308,
                "1756.87402397",
                "28.46694368",
                "0"
            ]
        ]
    },
    {
        "endpoint": "GET /api/v3/avgPrice",
        "description": "Current average price for a symbol",
        "parameters": [
            {"name": "symbol", "type": "STRING", "mandatory": "YES", "description": ""}
        ],
        "response": {
            "mins": 5,
            "price": "9.35751834"
        }
    },
    {
        "endpoint": "GET /api/v3/ticker/24hr",
        "description": "24hr ticker price change statistics",
        "parameters": [
            {"name": "symbol", "type": "STRING", "mandatory": "NO", "description": "Parameter symbol and symbols cannot be used in combination."},
            {"name": "symbols", "type": "STRING", "mandatory": "NO", "description": "Examples of accepted format for the symbols parameter: [\"BTCUSDT\",\"BNBUSDT\"] or %5B%22BTCUSDT%22,%22BNBUSDT%22%5D"},
            {"name": "type", "type": "ENUM", "mandatory": "NO", "description": "Supported values: FULL or MINI. If none provided, the default is FULL"}
        ],
        "response": {
            "symbol": "BNBBTC",
            "priceChange": "-94.99999800",
            "priceChangePercent": "-95.960",
            "weightedAvgPrice": "0.29628482",
            "prevClosePrice": "0.10002000",
            "lastPrice": "4.00000200",
            "lastQty": "200.00000000",
            "bidPrice": "4.00000000",
            "bidQty": "100.00000000",
            "askPrice": "4.00000200",
            "askQty": "100.00000000",
            "openPrice": "99.00000000",
            "highPrice": "100.00000000",
            "lowPrice": "0.10000000",
            "volume": "8913.30000000",
            "quoteVolume": "15.30000000",
            "openTime": 1499783499040,
            "closeTime": 1499869899040,
            "firstId": 28385,
            "lastId": 28460,
            "count": 76
        }
    },
    {
        "endpoint": "GET /api/v3/ticker/price",
        "description": "Symbol price ticker",
        "parameters": [
            {"name": "symbol", "type": "STRING", "mandatory": "NO", "description": "Parameter symbol and symbols cannot be used in combination. If neither parameter is sent, prices for all symbols will be returned in an array."},
            {"name": "symbols", "type": "STRING", "mandatory": "NO", "description": "Examples of accepted format for the symbols parameter: [\"BTCUSDT\",\"BNBUSDT\"] or %5B%22BTCUSDT%22,%22BNBUSDT%22%5D"}
        ],
        "response": [
            {
                "symbol": "LTCBTC",
                "price": "4.00000200"
            },
            {
                "symbol": "ETHBTC",
                "price": "0.07946600"
            }
        ]
    },
    {
        "endpoint": "GET /api/v3/ticker/bookTicker",
        "description": "Symbol order book ticker",
        "parameters": [
            {"name": "symbol", "type": "STRING", "mandatory": "NO", "description": "Parameter symbol and symbols cannot be used in combination. If neither parameter is sent, bookTickers for all symbols will be returned in an array."},
            {"name": "symbols", "type": "STRING", "mandatory": "NO", "description": "Examples of accepted format for the symbols parameter: [\"BTCUSDT\",\"BNBUSDT\"] or %5B%22BTCUSDT%22,%22BNBUSDT%22%5D"}
        ],
        "response": [
            {
                "symbol": "LTCBTC",
                "bidPrice": "4.00000000",
                "bidQty": "431.00000000",
                "askPrice": "4.00000200",
                "askQty": "9.00000000"
            },
            {
                "symbol": "ETHBTC",
                "bidPrice": "0.07946700",
                "bidQty": "9.00000000",
                "askPrice": "100000.00000000",
                "askQty": "1000.00000000"
            }
        ]
    },
    {
        "endpoint": "GET /api/v3/ticker",
        "description": "Rolling window price change statistics",
        "parameters": [
            {"name": "symbol", "type": "STRING", "mandatory": "YES", "description": "Either symbol or symbols must be provided"},
            {"name": "symbols", "type": "STRING", "mandatory": "NO", "description": "Examples of accepted format for the symbols parameter: [\"BTCUSDT\",\"BNBUSDT\"] or %5B%22BTCUSDT%22,%22BNBUSDT%22%5D"},
            {"name": "windowSize", "type": "ENUM", "mandatory": "NO", "description": "Defaults to 1d if no parameter provided. Supported windowSize values: 1m,2m....59m for minutes; 1h, 2h....23h - for hours; 1d...7d - for days. Units cannot be combined (e.g. 1d2h is not allowed)"},
            {"name": "type", "type": "ENUM", "mandatory": "NO", "description": "Supported values: FULL or MINI. If none provided, the default is FULL"}
        ],
        "response": [
            {
                "symbol": "BTCUSDT",
                "priceChange": "-154.13000000",
                "priceChangePercent": "-0.740",
                "weightedAvgPrice": "20677.46305250",
                "openPrice": "20825.27000000",
                "highPrice": "20972.46000000",
                "lowPrice": "20327.92000000",
                "lastPrice": "20671.14000000",
                "volume": "72.65112300",
                "quoteVolume": "1502240.91155513",
                "openTime": 1655432400000,
                "closeTime": 1655446835460,
                "firstId": 11147809,
                "lastId": 11149775,
                "count": 1967
            },
            {
                "symbol": "BNBBTC",
                "priceChange": "0.00008530",
                "priceChangePercent": "0.823",
                "weightedAvgPrice": "0.01043129",
                "openPrice": "0.01036170",
                "highPrice": "0.01049850",
                "lowPrice": "0.01033870",
                "lastPrice": "0.01044700",
                "volume": "166.67000000",
                "quoteVolume": "1.73858301",
                "openTime": 1655432400000,
                "closeTime": 1655446835460,
                "firstId": 2351674,
                "lastId": 2352034,
                "count": 361
            }
        ]
    },
    {
        "endpoint": "GET /api/v3/ticker/tradingDay",
        "description": "Trading Day Ticker",
        "parameters": [
            {"name": "symbol", "type": "STRING", "mandatory": "YES", "description": "Either symbol or symbols must be provided"},
            {"name": "symbols", "type": "STRING", "mandatory": "NO", "description": "Examples of accepted format for the symbols parameter: [\"BTCUSDT\",\"BNBUSDT\"] or %5B%22BTCUSDT%22,%22BNBUSDT%22%5D"},
            {"name": "timeZone", "type": "STRING", "mandatory": "NO", "description": "Default: 0 (UTC). Supported values for timeZone: Hours and minutes (e.g. -1:00, 05:45); Only hours (e.g. 0, 8, 4)"},
            {"name": "type", "type": "ENUM", "mandatory": "NO", "description": "Supported values: FULL or MINI. If none provided, the default is FULL"}
        ],
        "response": [
            {
                "symbol": "BTCUSDT",
                "priceChange": "-83.13000000",
                "priceChangePercent": "-0.317",
                "weightedAvgPrice": "26234.58803036",
                "openPrice": "26304.80000000",
                "highPrice": "26397.46000000",
                "lowPrice": "26088.34000000",
                "lastPrice": "26221.67000000",
                "volume": "18495.35066000",
                "quoteVolume": "485217905.04210480",
                "openTime": 1695686400000,
                "closeTime": 1695772799999,
                "firstId": 3220151555,
                "lastId": 3220849281,
                "count": 697727
            },
            {
                "symbol": "BNBUSDT",
                "priceChange": "2.60000000",
                "priceChangePercent": "1.238",
                "weightedAvgPrice": "211.92276958",
                "openPrice": "210.00000000",
                "highPrice": "213.70000000",
                "lowPrice": "209.70000000",
                "lastPrice": "212.60000000",
                "volume": "280709.58900000",
                "quoteVolume": "59488753.54750000",
                "openTime": 1695686400000,
                "closeTime": 1695772799999,
                "firstId": 672397461,
                "lastId": 672496158,
                "count": 98698
            }
        ]
    }
]

# Filepath to save the CSV file
csv_filepath = "binance_api_endpoints.csv"

# Write the API endpoints and descriptions to a CSV file
with open(csv_filepath, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["endpoint", "description", "parameters", "response"])
    writer.writeheader()
    for endpoint in binance_api_endpoints:
        writer.writerow({
            "endpoint": endpoint["endpoint"],
            "description": endpoint["description"],
            "parameters": str(endpoint["parameters"]),
            "response": str(endpoint["response"])
        })

print(f"CSV file created: {csv_filepath}")