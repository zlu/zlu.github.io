---
layout: post
title: "Taming Postgres on Heroku"
date: 2016-03-02
comments: true
categories: [featured, heroku, postgres, performance]
---

So you deployed an app on Heroku, because it’s easy and cheap to start with. After a while, your app is gaining traction and user base grows. Suddenly you find a slow app. There are many ways to tackle the performance issue. Making web pages more ajaxy and preload content, page caching, leveraging cdns, etc. In this blog I would like to focus on addressing some common database performance issues from outside in. Supposed we haven’t done much about tuning the database, and suppose we listened to Heroku and chose one of their posgres db instances to store our data.

A well-designed application serves 99% query from cache.

```bash
$ heroku pg:cache-hit -a my_database
      name      |         ratio          
      ----------------+------------------------
      index hit rate | 0.76517846526624000529
      table hit rate | 0.93700270695348766263
      (2 rows)
```

The command above shows that cache hit rate is low because heuristic tells us that cache hit for postgres should be above 99% to be considered performant. Heroku pg provides a command that tells you what the outliners are. Suppose we already know what the outliner is, then we need to issue an explain analyze against query under suspicion.
```bash
=> explain analyze SELECT COUNT(*) FROM "follows" WHERE "follows"."account_id" = 1 AND "follows"."active" = true; 
QUERY PLAN
---------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
Aggregate (cost=41274.28..41274.28 rows=1 width=0) (actual time=473.756..473.758 rows=1 loops=1) 
-> Index Scan using index_follows_on_follower_uid_and_account_id on follows (cost=0.09..41274.04 rows=466 width=0) (actual time=3.339..473.294 rows=333 loops=1) 
   Index Cond: (account_id = 1) 
   Filter: active 
   Rows Removed by Filter: 1 
   Total runtime: 474.043 ms 
   (6 rows)
```
The result explains to us that we are performing a query that takes longer than half a second to complete. That’s quite slow. It’s good that the db is performing an index scan since we are looking at account_id and active columns. However the index used is wrong. So let’s add a better index:
```bash
add_index :follows, [:account_id, :active]
```
Now if we try the same query:
```bash
=> explain analyze SELECT COUNT(*) FROM "follows" WHERE "follows"."account_id" = 1 AND "follows"."active" = true;
QUERY PLAN
--------------------------------------------------------------------------------------------------------------------------------------------------------------- 
Aggregate (cost=679.66..679.66 rows=1 width=0) (actual time=0.022..0.023 rows=1 loops=1) 
-> Index Only Scan using index_follows_on_account_id_and_active on follows (cost=0.09..679.43 rows=466 width=0) (actual time=0.017..0.017 rows=0 loops=1) 
   Index Cond: ((account_id = 1) AND (active = true)) 
   Filter: active 
   Heap Fetches: 0 
   Total runtime: 0.064 ms 
   (6 rows) 
```
Total runtime has dropped down to a fraction of 1 millisecond from close to half a second. This shows our measure, deploy, and measure approach of fixing db performance issues driven by outside-in approach.

Let’s say you iteratively issue pg:outliers and introduce approapriate indexes but then the performance doesn’t seem to improve any more. This is happening because your fine-tuned app is getting even more attraction. More data get stored and it is probably time to evaluate your database needs. A good place to start is pg:info command:
```
Table size: 16.5 GB
pg:total_index_size: 1022 MB
```
Suppose you are using Heroku’s Standard0 plan: - Cache: 1GB - Storage: 64 GB - Conn limit: 120

Ah now the problem is that your index is too big to fit into cache. There are two things we can do. First thing is getting rid of unused index. Sometimes developers can get index happy and keep adding them, but it’s easy to forget to change or remove obsolete indexes when db schema changes.
```bash
#heroku pg:diagnose -a my_database 
...
Never Used Indexes public.follows::index_follows_on_follower_uid_and_account_id 0.00 0.00 104 MB 230 MB
...
```
`pg:diagnose` shows you what indexes are not being used. Since index negatively affect insertion/update/deletion, unused indexes should be removed. The added benefit is that we are also saving precious db storage. If used indexes still doesn’t fit into cache, then it’s time to upgrade database. Some heuristics for db performance for web applications: - Very common queries returning small data set: ~ 1ms - Occasionally run queries returning small data set: ~ 5ms - Common query returning larger data set: ~ 10ms - Uncommon queries returning larger data set: ~ 100ms

Conditional OR composite index. A conditional would be where only current = true, where as the composite would index both values. A conditional is commonly more valuable when you have a smaller set of what the values may be, meanwhile the composite is when you have a high variability of values.

Now is a good time to mention index types.
### Common Index Types

B-Tree, this is the default index postgres creates:

    used for both equality and range queries
    operated against all data types

Hash Index:

    useful only for equality comparisons
    not transaction safe
    needs to be manually rebuilt after crashes

Partial Indexes

    Index with a where clause
    Increased scan speed by reducing index size
    Commonly applied to boolean field for the where clause

Expression Indexes

    Useful for queries that match on some function or modification of data, such as case-insensitive comparison of email login

create index users_lower_email on users(lower(email));

Last words about sequential scan v.s. index scan

    Index scan requires more IO than sequential scan

    Sequential scan is faster when result set is relatively large (5-10% of all rows in the table)

