-- Fetch the last completed minute of raw metrics in the same format as the
-- data CSV (datetime, module, tags, metrics) so the rows can be appended
-- directly to the CSV that the EWMA monitor reads.
--
-- Returns one row per module (zbx_cpu, zbx_memory, zbx_system) for the last
-- fully completed minute, matching the CSV layout that cleaning_utils.py
-- expects.

SELECT
  datetime::text                 AS datetime,
  module,
  COALESCE(tags::text, '{}')     AS tags,
  metrics::text                  AS metrics
FROM metrics.minute
WHERE nodename = 'ip-10-175-137-168'
  AND module IN ('zbx_cpu', 'zbx_memory', 'zbx_system')
  -- last fully completed minute (not the current, still-running minute)
  AND datetime >= date_trunc('minute', now() - interval '1 minute')
  AND datetime <  date_trunc('minute', now())
ORDER BY datetime, module;
