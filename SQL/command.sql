SELECT
  date_trunc('minute', datetime::timestamptz) AS datetime,
  
  -- CPU
  MAX((metrics->>'user.pct')::float)        FILTER (WHERE module = 'zbx_cpu')    AS cpu_user_pct,
  MAX((metrics->>'system.pct')::float)      FILTER (WHERE module = 'zbx_cpu')    AS cpu_system_pct,
  MAX((metrics->>'iowait.pct')::float)      FILTER (WHERE module = 'zbx_cpu')    AS cpu_iowait_pct,
  MAX((metrics->>'switches')::float)        FILTER (WHERE module = 'zbx_cpu')    AS cpu_switches,
  MAX((metrics->>'interrupts')::float)      FILTER (WHERE module = 'zbx_cpu')    AS cpu_interrupts,
  
  -- Memory
  MAX((metrics->>'util.pct')::float)             FILTER (WHERE module = 'zbx_memory') AS mem_util_pct,
  MAX((metrics->>'committed_as.kbytes')::float)  FILTER (WHERE module = 'zbx_memory') AS mem_committed_as_kbytes,
  
  -- System
  MAX((metrics->>'load_avg_1')::float)      FILTER (WHERE module = 'zbx_system') AS sys_load_avg_1,
  MAX((metrics->>'load_avg_15')::float)     FILTER (WHERE module = 'zbx_system') AS sys_load_avg_15,
  MAX((metrics->>'proc_count')::float)      FILTER (WHERE module = 'zbx_system') AS sys_proc_count,
  MAX((metrics->>'swap_used.pct')::float)   FILTER (WHERE module = 'zbx_system') AS sys_swap_used_pct

FROM metrics.minute
WHERE nodename = 'ip-10-175-137-168'
  AND module IN ('zbx_cpu', 'zbx_memory', 'zbx_system')
  -- Filtert exakt die letzte vollendete Minute:
  AND datetime >= date_trunc('minute', now() - interval '1 minute')
  AND datetime <  date_trunc('minute', now())
GROUP BY 1;