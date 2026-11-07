---
layout: post
title: "A tiny HTTP server in Python: one socket, then many threads"
description: "I wrote a small TCP web server in Python: one request at a time, then a multithreaded version, plus a looping GET client that hammers it without a browser."
keywords: "Python sockets, TCP, HTTP, web server, multithreading, AF_INET, SOCK_STREAM, listen backlog, daemon threads"
date: 2026-11-07
comments: true
tags:
  - networking
  - python
  - sockets
  - TCP
  - HTTP
---

I wanted a clearer picture of what a “web server” is doing under the hood, so I wrote a tiny one in Python. At the wire it is mostly: open a TCP socket, wait for a connection, read a few lines that start with `GET`, and write back a status line plus some HTML.

Below is what I ended up with: a **single-threaded** server that handles one client at a time, a **multithreaded** version that keeps accepting while another request is still writing, and a **looping client** that fires several GETs at once so the threading is visible without Chrome.


## What TCP is doing here

Browsers speak HTTP over TCP. TCP is connection-oriented and reliable: you `connect`, you get an ordered byte stream, and you close when you are done. For this toy server that is enough. No UDP packet reassembly, no TLS, just sockets and strings.

HTTP is the text on that stream: `GET /index.html …`, then `200 OK` or `404`.


## Single-threaded server

Core loop:

1. Create `socket(AF_INET, SOCK_STREAM)` — IPv4 + TCP.
2. `bind(('', port))` — listen on all interfaces.
3. `listen(1)` — backlog of one pending connection.
4. Forever: `accept()` → `recv` the request → open the file → send headers + body → `close`.

Happy path:

```python
from socket import *
import sys

serverSocket = socket(AF_INET, SOCK_STREAM)
serverPort = 6789
serverSocket.bind(('', serverPort))
serverSocket.listen(1)

while True:
    print('Ready to serve...')
    connectionSocket, addr = serverSocket.accept()
    try:
        message = connectionSocket.recv(1024).decode()
        filename = message.split()[1]          # e.g. /index.html
        with open(filename[1:]) as f:          # strip leading /
            outputdata = f.read()

        connectionSocket.send('HTTP/1.1 200 OK\r\n'.encode())
        connectionSocket.send('Content-Type: text/html\r\n\r\n'.encode())
        for ch in outputdata:
            connectionSocket.send(ch.encode())
        connectionSocket.send('\r\n'.encode())
        connectionSocket.close()
    except IOError:
        connectionSocket.send('HTTP/1.1 404 Not Found\r\n'.encode())
        connectionSocket.send('Content-Type: text/html\r\n\r\n'.encode())
        connectionSocket.send(
            b'<html><body><h1>404 Not Found</h1></body></html>\r\n'
        )
        connectionSocket.close()
```

What bit me hardest was the blank line after the headers.

HTTP ends the header block with a blank line: `\r\n\r\n`. I started the 404 path with a single `\r\n` after `Content-Type`:

```python
# wrong: only one \r\n. Browser keeps waiting for more headers.
connectionSocket.send('Content-Type: text/html\r\n'.encode())
connectionSocket.send(b'<html>…</html>\r\n')
```

The status line went out. The HTML went out. Chrome still showed a blank failure instead of my 404 page. From the browser’s point of view the body never started; the next bytes looked like more header text. Adding the second `\r\n` fixed it:

```python
connectionSocket.send('Content-Type: text/html\r\n\r\n'.encode())
```

Same rule on the 200 path. Same rule in the client request (`Host` / `Connection`, then `\r\n` alone). I kept forgetting that the “empty line” is part of the protocol, not decoration.

Two smaller notes:

- **`listen(1)` vs `listen(5)`**. The number is the OS *backlog*: how many completed (or half-open) connections can wait for `accept()`. It is not “max clients forever.” One is fine when I finish a request before accepting the next.
- **Parsing**. A real browser request has many headers. Here `message.split()[1]` is enough to grab the path from `GET /file HTTP/1.1`.

Limitation: while the process is reading a file or writing the response, nobody else gets `accept()`. Two tabs at once feel fine on localhost. Under any real load this is a queue of one.

I drop an `index.html` next to the script and open:

```text
http://127.0.0.1:6789/index.html
http://127.0.0.1:6789/nope.html
```

First renders the page. Second shows the 404 body (once the blank line is there).


## Multithreaded server

Same HTTP logic, moved into a worker:

- **Main thread** only listens and accepts on the fixed port.
- Each `accept()` returns a **new** connected socket (the server side uses an ephemeral port for that connection).
- Spawn a `Thread` that owns that socket for the request/response pair.

I also sleep for a second inside the worker. Localhost GETs are otherwise too fast; without that pause the overlapping work is hard to see in the log.

```python
from socket import *
from threading import Thread, current_thread
import sys
import time

def handle_client(connectionSocket, addr):
    name = current_thread().name
    try:
        message = connectionSocket.recv(1024).decode()
        if not message:
            return
        filename = message.split()[1]
        print(f'{name} handling {addr} path={filename}')
        time.sleep(1)  # stretch the request so concurrency shows up
        with open(filename[1:]) as f:
            outputdata = f.read()
        connectionSocket.send(b'HTTP/1.1 200 OK\r\n')
        connectionSocket.send(b'Content-Type: text/html\r\n\r\n')
        connectionSocket.send(outputdata.encode())
        connectionSocket.send(b'\r\n')
    except IOError:
        connectionSocket.send(b'HTTP/1.1 404 Not Found\r\n')
        connectionSocket.send(b'Content-Type: text/html\r\n\r\n')
        connectionSocket.send(b'<html><body><h1>404 Not Found</h1></body></html>\r\n')
    finally:
        print(f'{name} done {addr}')
        connectionSocket.close()

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 6789

serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
serverSocket.bind((SERVER_HOST, SERVER_PORT))
serverSocket.listen(5)   # small burst of pending connects while threads start
print(f'Listening on http://{SERVER_HOST}:{SERVER_PORT}/')

try:
    while True:
        connectionSocket, addr = serverSocket.accept()
        print(f'Connection from {addr}')
        Thread(
            target=handle_client,
            args=(connectionSocket, addr),
            daemon=True,
        ).start()
except KeyboardInterrupt:
    print('\nShutting down...')
finally:
    serverSocket.close()
    sys.exit()
```

How the knobs differ from the single-thread version:

| Knob | Single-thread | Multi-thread |
|:-----|:--------------|:-------------|
| `listen(n)` | `1` is enough | `5` (or more) so a burst can queue while workers start |
| `SO_REUSEADDR` | optional | handy when I Ctrl+C and restart on the same port |
| concurrency | one request in flight | one thread per connection |
| `daemon=True` | n/a | process can exit without joining every worker |

`daemon=True` means client threads do not keep the process alive after the main thread exits. Fine for this scratch server. A real one would shut down more carefully.

```text
[main] listen 127.0.0.1:6789
   │ accept → conn₁ ──► Thread₁ handle_client (sleep 1s)
   │ accept → conn₂ ──► Thread₂ handle_client (sleep 1s)
   └ accept → conn₃ ──► Thread₃ handle_client (sleep 1s)
```

Each pair is its own TCP connection. The listening socket never does `recv`/`send` of HTTP. The workers do.


## A looping GET client (no browser)

Browsers hide the request. I wanted bytes I can print, and a burst of concurrent connects so the threaded server has something to show. Host, port, and path are hardcoded. No argv.

```python
from socket import *
from threading import Thread
import time

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 6789
FILENAME = 'index.html'
N_REQUESTS = 8

def one_get(i):
    request = (
        f'GET /{FILENAME} HTTP/1.1\r\n'
        f'Host: {SERVER_HOST}:{SERVER_PORT}\r\n'
        f'Connection: close\r\n'
        f'\r\n'
    )
    client_socket = socket(AF_INET, SOCK_STREAM)
    try:
        client_socket.connect((SERVER_HOST, SERVER_PORT))
        client_socket.send(request.encode())
        response = b''
        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            response += chunk
        status = response.split(b'\r\n', 1)[0].decode(errors='replace')
        print(f'[{i}] {status} ({len(response)} bytes)')
    except Exception as e:
        print(f'[{i}] Error: {e}')
    finally:
        client_socket.close()

# Fire N connects at once; join so the process waits for all replies.
threads = []
t0 = time.time()
for i in range(N_REQUESTS):
    t = Thread(target=one_get, args=(i,))
    t.start()
    threads.append(t)
for t in threads:
    t.join()
print(f'done in {time.time() - t0:.2f}s')
```

Notes from wiring this up:

- **Hardcoded `SERVER_HOST` / `SERVER_PORT` / `FILENAME`**. Same constants as the server. I was not building a CLI; I was proving concurrency.
- **`AF_INET` / `SOCK_STREAM`**. Same pair as the server. Both ends of a TCP conversation.
- **`Connection: close`**. Ask the server to close after one response so each client’s `recv` loop ends when the peer closes.
- **One thread per request**. A serial `for` loop of connects would still work, but the wall clock would look like `N * 1s` because of the server sleep. Spawning the requests together finishes near one sleep, and the server log interleaves `Thread-… handling` lines.

Against the threaded server:

```bash
# terminal 1
python MultithreadedWebServer.py

# terminal 2
python client.py
```

Server side looks roughly like this (order varies):

```text
Listening on http://127.0.0.1:6789/
Connection from ('127.0.0.1', 54321)
Connection from ('127.0.0.1', 54322)
Thread-1 handling ('127.0.0.1', 54321) path=/index.html
Thread-2 handling ('127.0.0.1', 54322) path=/index.html
…
done in ~1s on the client, not ~8s
```

That overlap is the point of the threads. Point the same client at the single-threaded server and the client wall clock stretches toward `N` seconds.


## What stuck

1. HTTP over sockets is readable text on a TCP byte stream. The blank line (`\r\n\r\n`) after headers is required; without it the browser never treats the HTML as a body.
2. The single-threaded version made `bind` / `listen` / `accept` / `recv` / `send` obvious without concurrency noise.
3. Threads do not replace TCP. They only let the accept loop keep moving while another connection is mid-response. A looping concurrent client makes that visible in wall-clock time.
4. A hand-rolled client forces the GET line into view. The browser’s “it just works” hides a lot of bugs.

Still a toy: no persistent connections, no `Content-Length`, no path safety (`../`), no HTTPS. That is fine. I wanted the socket lifecycle, not nginx.


## Run checklist

```bash
# Single-thread (browser OK)
python WebServer.py
# → http://127.0.0.1:6789/index.html

# Multi-thread + concurrent client loop
python MultithreadedWebServer.py   # terminal 1
python client.py                   # terminal 2; 8 GETs at once
```

I may later add a short Wireshark filter (`tcp.port == 6789`) so the three-way handshake and the ASCII GET show up next to the code.
