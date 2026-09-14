#!/usr/bin/env node

const { Pool } = require("pg");

const GOOGLE_API_TOKEN = ""; // Fill in, or set GOOGLE_API_TOKEN.
const GOOGLE_MODEL = "gemini-3.8-flash"; // Fill in, or set GOOGLE_MODEL (for example, gemini-2.0-flash).
const BATCH_SIZE = 32;
const CONCURRENCY = 4;
const POSTGRES_USER = ""; // Fill in, or set POSTGRES_USER.
const POSTGRES_PASSWORD = ""; // Fill in, or set POSTGRES_PASSWORD.

const pool = new Pool({
	host: process.env.POSTGRES_HOST || "localhost",
	port: Number(process.env.POSTGRES_PORT || 5432),
	database: "news_trading_window",
	user: process.env.POSTGRES_USER || POSTGRES_USER,
	password: process.env.POSTGRES_PASSWORD || POSTGRES_PASSWORD,
});

async function getSentiment(text) {
	const token = process.env.GOOGLE_API_TOKEN || GOOGLE_API_TOKEN;
	const model = process.env.GOOGLE_MODEL || GOOGLE_MODEL;
	if (!token || !model) throw new Error("Set GOOGLE_API_TOKEN and GOOGLE_MODEL.");

	const response = await fetch(
		`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(token)}`,
		{
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({
				contents: [{ parts: [{ text: [
					"Classify the sentiment and reply with only one number.",
					"1=Strongly negative, 2=negative, 3=neutral, 4=positive, 5=Strongly positive.",
					"Return only a single number, no other text or punctuation.",
					"Text:", text,
				].join("\n") }] }],
				generationConfig: { temperature: 0, maxOutputTokens: 1024 },
			}),
		},
	);
	if (!response.ok) throw new Error(`Google AI ${response.status}: ${await response.text()}`);
	const body = await response.json();
	const match = body.candidates?.[0]?.content?.parts?.[0]?.text?.trim().match(/^[1-5]$/);
	if (!match) {
		console.error("Google AI response:", JSON.stringify(body, null, 2));
		return NULL; // Return 0 for invalid sentiment.
	}
	return Number(match[0]);
}

async function listModels() {
	const token = process.env.GOOGLE_API_TOKEN || GOOGLE_API_TOKEN;
	if (!token) throw new Error("Set GOOGLE_API_TOKEN.");

	const response = await fetch(
		`https://generativelanguage.googleapis.com/v1beta/models?key=${encodeURIComponent(token)}`,
	);
	if (!response.ok) throw new Error(`Google AI ${response.status}: ${await response.text()}`);
	return response.json();
}

async function main() {
	const client = await pool.connect();
	try {
		// PostgreSQL has no uint8 type; SMALLINT with this constraint provides the same range.
		await client.query(`
			ALTER TABLE public.articles
			ADD COLUMN IF NOT EXISTS sentiment SMALLINT
			CHECK (sentiment IS NULL OR sentiment BETWEEN 1 AND 5)
		`);

		// ctid is used because the supplied query does not expose a primary key.
		const { rows } = await client.query(`
			SELECT ai.ctid::text AS row_id, ai.consolidated_insights
			FROM public.articles AS ai
			WHERE ai.consolidated_insights IS NOT NULL AND ai.sentiment IS NULL
		`);

		console.log(`Found ${rows.length} articles to process.`);

		for (let offset = 0; offset < rows.length; offset += BATCH_SIZE * CONCURRENCY) {
			const chunks = [];
			for (let i = offset; i < Math.min(offset + BATCH_SIZE * CONCURRENCY, rows.length); i += BATCH_SIZE) {
				chunks.push(rows.slice(i, i + BATCH_SIZE));
			}
			await Promise.all(chunks.map(async (chunk) => {
				const sentiments = await Promise.all(chunk.map((row) => {
					const text = String(row.consolidated_insights).split(/\r?\n/).slice(1).join("\n");
					return getSentiment(text);
				}));
				await client.query("BEGIN");
				try {
					for (let i = 0; i < chunk.length; i++) {
						await client.query(
							"UPDATE public.articles SET sentiment = $1 WHERE ctid = $2::tid",
							[sentiments[i], chunk[i].row_id],
						);
					}
					await client.query("COMMIT");
				} catch (error) {
					await client.query("ROLLBACK");
					throw error;
				}
			}));
			console.log(`Processed ${Math.min(offset + BATCH_SIZE * CONCURRENCY, rows.length)}/${rows.length}`);
		}
	} finally {
		client.release();
		await pool.end();
	}
}

main().catch((error) => {
	console.error(error);
	process.exitCode = 1;
});
