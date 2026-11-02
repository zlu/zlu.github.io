-- zlu.me web visitor funnel for the BetterReads acquisition monitor.
-- Run once in the shared BetterReads Supabase SQL Editor.
--
-- Ingest: betterreads-fe PUT /api/zlume-web-stats (CORS from zlu.me)
-- Admin:  betterreads-fe /admin/analytics?app=zlume

create table if not exists public.zlume_web_visitor_stats (
  visitor_id text primary key,
  first_seen_at timestamptz not null,
  last_seen_at timestamptz not null default now(),
  platform text not null default 'web',
  payload jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

create index if not exists idx_zlume_web_visitor_stats_last_seen
  on public.zlume_web_visitor_stats (last_seen_at desc);

create or replace view public.zlume_web_visitor_stats_human as
select *
from public.zlume_web_visitor_stats
where coalesce((payload->>'likelyBot')::boolean, false) = false;

alter table public.zlume_web_visitor_stats enable row level security;

-- No anon policies — service role (Worker upsert + BR admin) only.
revoke all on public.zlume_web_visitor_stats from anon, authenticated;
grant select, insert, update on public.zlume_web_visitor_stats to service_role;
grant select on public.zlume_web_visitor_stats_human to service_role;

create or replace function public.zlume_upsert_web_visitor_stats(
  p_visitor_id text,
  p_first_seen_at timestamptz,
  p_last_seen_at timestamptz,
  p_platform text,
  p_payload jsonb
)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into zlume_web_visitor_stats (
    visitor_id,
    first_seen_at,
    last_seen_at,
    platform,
    payload,
    updated_at
  )
  values (
    p_visitor_id,
    p_first_seen_at,
    p_last_seen_at,
    coalesce(nullif(trim(p_platform), ''), 'web'),
    coalesce(p_payload, '{}'::jsonb),
    now()
  )
  on conflict (visitor_id) do update set
    first_seen_at = zlume_web_visitor_stats.first_seen_at,
    last_seen_at = excluded.last_seen_at,
    platform = excluded.platform,
    payload = excluded.payload,
    updated_at = now();
end;
$$;

revoke all on function public.zlume_upsert_web_visitor_stats(text, timestamptz, timestamptz, text, jsonb)
  from public, anon, authenticated;
grant execute on function public.zlume_upsert_web_visitor_stats(text, timestamptz, timestamptz, text, jsonb)
  to service_role;
