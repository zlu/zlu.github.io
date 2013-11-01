---
layout: post
title: "A Small Difference Between Mongoid and ActiveRecord Query Interface"
date: 2012-10-16 21:29
comments: true
categories: [MongoDB, ActiveRecord, SQL]
---

There are definitely some gotchas when switching from ActiveRecord to MongoDB (with Mongoid for example).  One of them
is how the IN operator in SQL is supported.

In ActiveRecord's Query Interface, we can do `MyModel.where(:id => [1,3,5])` and expect this SQL query to be generated:
`select * from MyModel where (MyModel.id IN (1,3,5))`.  Not true when it comes to Mongoid.

Mongo supports quite a few conditional operators such as < (or $lt), $all, $in, etc.  So the above query needs to be constructed
as such:

`MyModel.where(:id => {:$in => [1,3,5]})`

Another example:

`MyModel.where(:created_at => {:$gt => 2.days.ago})`

This builds a query that returns instances of MyModel that were created in the past 2 days.

#TODO ActiveRecord#from - does not exist in Mongoid

