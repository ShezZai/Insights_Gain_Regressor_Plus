require('dotenv').config();
const { Client } = require('pg');
const axios = require('axios');

// Twelve Data Free Tier Limits: 8 requests per minute
const TWELVE_DATA_API_KEY = process.env.TWELVE_DATA_API_KEY || '6ead19b8833840a086201a06303ca429';
const API_DELAY_MS = 8000; // 8 seconds per request to stay under 8 req/min

const dbConfig = {
    user: 'nivnoach',
    host: 'localhost',
    database: 'news_trading_window',
    password: process.env.DB_PASSWORD,
    port: 5432,
};

// Helper: Sleep to respect rate limits
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Helper: Find the closest available trading data point
function getNearestPrice(timeSeries, targetTimeMs) {
    if (!timeSeries || timeSeries.length === 0) return null;
    
    // timeSeries from Twelve Data is usually descending (newest first)
    // Find the record with the minimum absolute time difference
    let closestRecord = timeSeries[0];
    let minDiff = Math.abs(new Date(closestRecord.datetime).getTime() - targetTimeMs);

    for (const record of timeSeries) {
        const recordTime = new Date(record.datetime).getTime();
        const diff = Math.abs(recordTime - targetTimeMs);
        if (diff < minDiff) {
            minDiff = diff;
            closestRecord = record;
        }
    }
    return parseFloat(closestRecord.close);
}

async function main() {
    const client = new Client(dbConfig);
    await client.connect();
    console.log('Connected to PostgreSQL local database.');

    try {
        // 1. Ensure columns exist (assuming table name is 'news_articles')
        // UPDATE THIS TABLE NAME IF DIFFERENT
        const tableName = 'articles';
        await client.query(`
            ALTER TABLE ${tableName}
            ADD COLUMN IF NOT EXISTS gain_12h_after_article NUMERIC,
            ADD COLUMN IF NOT EXISTS gain_24h_after_article NUMERIC,
            ADD COLUMN IF NOT EXISTS gain_36h_after_article NUMERIC,
            ADD COLUMN IF NOT EXISTS gain_48h_after_article NUMERIC;
        `);

        // 2. Fetch target data: id, first ticker (Postgres arrays are 1-indexed), and UTC date
        const res = await client.query(`
            SELECT id, tickers[1] AS ticker_first, published_utc 
            FROM ${tableName} 
            WHERE tickers IS NOT NULL AND array_length(tickers, 1) > 0
        `);
        const rows = res.rows;
        
        // 3. Find unique tickers and their date boundaries
        const tickerMap = {};
        for (const row of rows) {
            const ticker = row.ticker_first;
            const pubDate = new Date(row.published_utc).getTime();
            
            if (!tickerMap[ticker]) {
                tickerMap[ticker] = { min: pubDate, max: pubDate };
            } else {
                if (pubDate < tickerMap[ticker].min) tickerMap[ticker].min = pubDate;
                if (pubDate > tickerMap[ticker].max) tickerMap[ticker].max = pubDate;
            }
        }

        const uniqueTickers = Object.keys(tickerMap);
        console.log(`Found ${uniqueTickers.length} unique tickers across ${rows.length} rows.`);

        // 4. Fetch Time Series from Twelve Data
        const tickerDataCache = {};
        for (let i = 0; i < uniqueTickers.length; i++) {
            const ticker = uniqueTickers[i];
            
            // Format dates for Twelve Data (YYYY-MM-DD)
            // Add a 3-day buffer to the max date to cover the +48h period and weekends
            const startDateStr = new Date(tickerMap[ticker].min - (24 * 60 * 60 * 1000)).toISOString().split('T')[0];
            const endDateStr = new Date(tickerMap[ticker].max + (3 * 24 * 60 * 60 * 1000)).toISOString().split('T')[0];
            
            console.log(`Fetching data for ${ticker} (${i + 1}/${uniqueTickers.length})...`);
            
            const url = `https://api.twelvedata.com/time_series?symbol=${ticker}&interval=1h&start_date=${startDateStr}&end_date=${endDateStr}&outputsize=5000&apikey=${TWELVE_DATA_API_KEY}`;
            
            try {
                const apiRes = await axios.get(url);
                if (apiRes.data.status === 'ok') {
                    tickerDataCache[ticker] = apiRes.data.values; // Array of { datetime, open, high, low, close, volume }
                } else {
                    console.error(`Twelve Data API Error for ${ticker}:`, apiRes.data.message);
                }
            } catch (err) {
                console.error(`Failed to fetch ${ticker}:`, err.message);
            }

            // Respect rate limits
            if (i < uniqueTickers.length - 1) {
                await sleep(API_DELAY_MS);
            }
        }

        // 5. Calculate Gains and Update DB
        console.log('Calculating gains and updating database...');
        
        // Use a transaction for bulk updates
        await client.query('BEGIN');
        
        for (const row of rows) {
            const ticker = row.ticker_first;
            const timeSeries = tickerDataCache[ticker];
            
            if (!timeSeries) continue; // Skip if API failed for this ticker
            
            const pubMs = new Date(row.published_utc).getTime();
            const hourMs = 60 * 60 * 1000;
            
            // Get nearest prices (handles weekend/overnight market closures)
            const price0 = getNearestPrice(timeSeries, pubMs);
            const price12 = getNearestPrice(timeSeries, pubMs + (12 * hourMs));
            const price24 = getNearestPrice(timeSeries, pubMs + (24 * hourMs));
            const price36 = getNearestPrice(timeSeries, pubMs + (36 * hourMs));
            const price48 = getNearestPrice(timeSeries, pubMs + (48 * hourMs));
            
            // Calculate gains if base price is valid and not zero
            if (price0) {
                const gain12 = price12 ? (price12 / price0) - 1 : null;
                const gain24 = price24 ? (price24 / price0) - 1 : null;
                const gain36 = price36 ? (price36 / price0) - 1 : null;
                const gain48 = price48 ? (price48 / price0) - 1 : null;

                // Parameterized update query
                const updateQuery = `
                    UPDATE ${tableName}
                    SET gain_12h_after_article = $1,
                        gain_24h_after_article = $2,
                        gain_36h_after_article = $3,
                        gain_48h_after_article = $4
                    WHERE id = $5
                `;
                
                await client.query(updateQuery, [gain12, gain24, gain36, gain48, row.id]);
            }
        }
        
        await client.query('COMMIT');
        console.log('Update complete.');

    } catch (error) {
        await client.query('ROLLBACK');
        console.error('Error during execution:', error);
    } finally {
        await client.end();
    }
}

main();