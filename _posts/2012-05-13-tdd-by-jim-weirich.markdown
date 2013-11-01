---
layout: post
title: "TDD by Jim Weirich"
date: 2012-05-13 18:28
comments: true
categories: [BDD, Agile]
---

I invited Jim Weirich to give us a week-long training on TDD/BDD/Agile.  Here're the raw notes.

The way of Testivus

Code driven from tests are more composable

RSpec - Focus on Behavior

Avoid testing state

As a client, I don't care about internal member variables or implementations, I only care about the result.

Reveal our intentions

Focus on the design

Forget unit, describe context

context is an alias to describe

describe/context is equivalent to creating a class of the object and behavior being described/contexted.
  class DescribeRing . . end
  class ContextWhenEmpty . . end
Also class inherits from top-level class
Why helper methods should be in various levels (nesting) of tests

predicate - any method ends with ?

be/equal => object identity

eql or == is structural equality

let(:symbol) { } creates a named method identified by :symbol.  When referred, the block will be executed exactly once.

Implicit subject

its(:symbol) {  should be_true }
specify is a 'kind of' its

Given/When/Then

rspec differentiate mocks and stubs.  stubs doesn't have to be called while mocks have to be at least once.
constraints are should_recieve, once, etc.
actions are and_return

stubs don't care how many times you call it or parameters passed in

Flatten the cost of change in software engineering - how?
  The key is to increase feedback

YAGNI - You ain't going to need it #Agile - No speculative coding!

What is simple?
  Passes all the tests
  Reveals the intention of the developer
  No Duplication
  Fewest huber of classes or methods

Collective Ownership

Sustainable Pace (40-hr work week in America)

Initial Importance
  Must/Should/Could/Won't
  Do high value and high risk stories first

