-- NYSE full closures and early closes (13:00 ET) covering 2024-11-01 .. 2026-05-30,
-- the published_utc range of this corpus.
CREATE TEMP TABLE nyse_holidays(d date) ON COMMIT DROP;
INSERT INTO nyse_holidays VALUES
  ('2024-11-28'),('2024-12-25'),
  ('2025-01-01'),('2025-01-09'),  -- 2025-01-09: national day of mourning (Carter)
  ('2025-01-20'),('2025-02-17'),('2025-04-18'),('2025-05-26'),('2025-06-19'),
  ('2025-07-04'),('2025-09-01'),('2025-11-27'),('2025-12-25'),
  ('2026-01-01'),('2026-01-19'),('2026-02-16'),('2026-04-03'),('2026-05-25');

CREATE TEMP TABLE nyse_early_close(d date) ON COMMIT DROP;  -- close 13:00 ET
INSERT INTO nyse_early_close VALUES
  ('2024-11-29'),('2024-12-24'),
  ('2025-07-03'),('2025-11-28'),('2025-12-24');

DELETE FROM articles
WHERE id IN (
  SELECT s.id
  FROM (SELECT id, (published_utc AT TIME ZONE 'America/New_York') AS et FROM articles) s
  WHERE s.et IS NULL
     OR NOT (
          extract(isodow FROM s.et) <= 5
      AND s.et::date NOT IN (SELECT d FROM nyse_holidays)
      AND s.et::time >= TIME '09:30'
      AND s.et::time <  (CASE WHEN s.et::date IN (SELECT d FROM nyse_early_close)
                              THEN TIME '11:30'   -- 13:00 close - 90 min
                              ELSE TIME '14:30'   -- 16:00 close - 90 min
                         END)
     )
);
