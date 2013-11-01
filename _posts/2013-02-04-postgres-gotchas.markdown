---
layout: post
title: "Postgres Gotchas"
date: 2013-02-04 11:16
comments: true
categories: [database, upgrade, sharedmemory]
---

I run postgres locally for development on OS X (mountain lion).  I've run into a few issues and am documenting solutions
here.

Issue 1: Insufficient Shared Memory
----

Postgres requires sufficient shared memory or it won't start.  If there isn't enough, postgres server will fail to start
and in server.log you will see this:

    FATAL:  could not create shared memory segment: Cannot allocate memory
    DETAIL:  Failed system call was shmget(key=5432001, size=3391488, 03600).
    HINT:  This error usually means that PostgreSQL's request for a shared memory segment exceeded available memory or swap space, or exceeded your kernel's SHMALL parameter.  You can either reduce the request size or reconfigure the kernel with larger SHMALL.  To reduce the request size (currently 3391488 bytes), reduce PostgreSQL's shared memory usage, perhaps by reducing shared_buffers or max_connections.

There are two ways to correct this problem.

1. Up the shared memory configuration.

    On mountain lion, you could create or modify your /etc/sysctl.conf so it has this:

        zlu@zlu-mba:~/projects/me/blog-zlu (master)$ cat /etc/sysctl.conf
        kern.sysv.shmmax=4194304
        kern.sysv.shmmin=1
        kern.sysv.shmmni=32
        kern.sysv.shmseg=8
        kern.sysv.shmall=1024

    Make sure shmmax is large enough for postgres and other software that requires shared memeory.
    To figure out how much shared memeory you will need for postgres, take a look at postgresql.conf (probably under /usr/local/var/postgres)

        max_connections = 20                    # (change requires restart)
        # Note:  Increasing max_connections costs ~400 bytes of shared memory per
        # connection slot, plus lock space (see max_locks_per_transaction).

    More details can be found here:
    [Upgrade Postgres](http://www.postgresql.org/docs/9.2/static/kernel-resources.html#SYSVIPC)

2. Reduce resource requirements
    In postgresql.conf you could search for "shared memory" and see all resources (such as max_connections and max_prepared_transactions).
    Reduce the requirements as you see fit.

Issue 2: Upgrading from 9.1.x to 9.2.x
----

If you need to migration database, you may run into path issues.

Here is the steps that work for me:

1. Backup **BEFORE** you install 9.2 (with homebrew).  This is important if you want to avoid the hussle later with two versions
    of <code>pg_ctl</code> stomping onto each other.  While the 9.1.x server is running, use <code>pg_dumpall</code> to export the database.
2. Stop 9.1.x server
3. mv the data directory (e.g. /usr/local/var/postgres) to postgres.bk.
4. Install postgres 9.2.x (with homebrew).
5. Create symlink for plist file for launchctl (it is named differently than 9.1 so you will need to do it again).  The post-install
    instruction of homebrew contains the exact command.
        cp /usr/local/Cellar/postgresql/9.2.1/homebrew.mxcl.postgresql.plist ~/Library/LaunchAgents/
    And remove the plist file that belonged to 9.1.x
6. <code>initdb</code> (this creates the 9.2.x version of the database).
7. Import the postgres data backup from the previous <code>pg_dumpall</code> command.
8. Load plist:
        launchctl load -w ~/Library/LaunchAgents/homebrew.mxcl.postgresql.plist

[Migrating Postgres](http://www.postgresql.org/docs/9.2/static/upgrading.html)

Issue 3: Rails Server Fails to Start
----

If you are doing Rails development, after upgrade postgres and issue 'rails s' for your project, you may see this error
which fails Rails server from starting:

    PGError (could not connect to server: Permission denied
        Is the server running locally and accepting
        connections on Unix domain socket "/var/pgsql_socket/.s.PGSQL.5432"?
    )

Since postgres 9.2, the unix socket file has changed to /var/pgsql_socket/alt.  The pg gem is compiled against the older version
of postgres (9.1.x) so it looks for the socket file in wrong location.  The easist thing to do is to uninstall and reinstall
pg gem.
