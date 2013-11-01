---
layout: post
title: "Rolling Deployment"
date: 2012-11-07 20:30
comments: true
categories: [Deployment, Apache, Capistrano]
---

It is common to deploy new code to production several times a week (or even a day) in an agile shop.  How to make deployment
less intrusive to end user has becoming a larger issue.  If you simply use `cap deploy production`, some web requests will
time out hence hamper consumer experience.  Heroku supports maintenance mode, which can be turned on before a deployment.
User will see a maintenance page instead of unresponsive web page.  This is ok but not ideal.

Martin Fowler wrote about [Blue Green Deployment](http://martinfowler.com/bliki/BlueGreenDeployment.html).  The canonical
form of this deployment strategy is to maintain two identical databases for each environment.  There is an issue of
dealing with missed transactions when deploying to one environment (stand-by) while the live environment is still taking
web requests.  There are a few ways to take care of this such as putting the live environment into read-only mode before the
cut-over.

For a small application (or a typical start-up scenario), having two database is a bit of over-kill and entails higher
operational costs.  With rolling deployment,  all the production web (application) servers share the same database.  To
reduce application downtime, only one web server will be taken out of the load balancer at a time.  This web server will then
be loaded with new code.  We can then run some quick automated tests (simple selenium tests for example) to ensure the build
is sane, against this web server.  If tests pass, we bring the node back into the load balancer.  We then repeat this process
for the next server.  If there are database changes that could potentially affect the state of the application and cause
inconsistency, then measure must be taken in the application to mitigate the problem through for example, back-fill.

Setup
----
A typical setup for a web app where multiple web servers (such as Unicorn) share the same database server.

<pre>
                  |-> web server1 (red)   -|
load balancer --> |-> web server2 (blue)  -|---> database
                  |-> web server3 (green) -|
</pre>

Steps
----

* Take the red web server out of load balancer
* Deploy new code to red
* Run deployment tests against red to ensure everything is sane
* If tests pass, put red back to load balancer and continue to the next node, blue
* If tests fail, roll back

Example
----

In this example, I use Apache's [mod_proxy_balancer](http://httpd.apache.org/docs/2.2/mod/mod_proxy_balancer.html).
I use a typical Rails web server setup and [Capistrano](https://github.com/capistrano/capistrano) for deployment.

{% highlight ruby %}
role :app, "server1.com", :group => :red
role :app, "server2.com", :group=> :blue
role :app, "server3.com", :group => :green
role :apache, "loadbalancer.com", :no_release => true
{% endhighlight %}

{% highlight ruby %}
def manage_node(action, group)
  uri = URI.parse("http://#{roles[:apache]}/balancer-manager")
  response_body = Net::HTTP.get_response(uri).body
  nonce = response_body.match(/nonce=(.*)\"/)[1]
  node = find_servers(:roles => :app, :only => {:group => group.to_sym})[0]
  params = {:w => "http://#{node}:8080", :b => stage, :nonce => nonce, :dw => action}
  uri.query = URI.encode_www_form(params)
  response = Net::HTTP.get_response(uri)
  response.code == "200"
end
{% endhighlight %}

mod_proxy_balancer has a balancer-manager GUI.  Admin is able to enable and disable node (web server) via a form.
manage_node method essentially submit the form to the balancer-manager.

To take red node out of the load balancer, simply define a task such as:
``` ruby
task :enable do
  manage_mode("Enable", group),
end
```
The value of group can be passed in as a command line option for Capistrano

Other Considerations
----

Why do I suggest running some basic tests as the acceptance criteria for rolling deployment?  I assume that the code
being deployed is well tested and passed CI/QA etc.  But there're all kinds of factors that can be different between test,
CI, and production environment.  Some library have different versions, VM images, or even the OS can be different.  I like
to use a Mac Mini for in-house CI (or Travis for open-source projects).  There's no guarantee that Travis or Mac Mini
can be kept up-to-date with production environment.  I could spin up a CI instance that replicates production but keeping
them in-sync still takes much effort.  There are also possibilities that integration test suites do not test views where
a simple route change could prevent user from successfully logging in.  By running a simple Selenium test to login and
perform one core function of the application, it gives me some level of assurance.

Capistrano supports deployment to a single server using HOSTS or HOSTFITLER command line options.  However, the load balancer
will keep sending requests to the server during deployment and server restart.

Note
----
[find_server API](http://rdoc.info/github/capistrano/capistrano/master/Capistrano/Configuration/Servers)