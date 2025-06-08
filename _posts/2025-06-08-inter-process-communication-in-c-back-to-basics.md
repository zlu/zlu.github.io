---
layout: post
title: 'Inter Process Communication in C: Back to Basics'
date: 2025-06-08 20:52 +0800
tags:
  - c
  - inter process communication
  - IPC
  - anonymous pipes
  - unix
  - linux
description: Interprocess communication (IPC) with anonymous pipes.
comments: true
---
Interprocess communication (IPC) refers to the mechanisms and methods that allow processes to communicate with each other and synchronize their execution.  By default, processes have separate memory spaces and don't share data.  In reality, processes often need to share data and resources.  In addition, they will need to coordinate their executions to achieve desired behavior.  This is called process synchronization.

Anonymous pipes are a form of the simplest IPC mechanism.  It allows one-way communication between related processes.  Typically, these are parent and child processes.  Data flows only in one direction (unidirection).  Only processes with a common ancestor can use this mechanism.  Its existance is dependent upon the parent process.  It is also memory-based because it uses a buffer in kernel memory.  Concretely speaking, an anonymous pipe is created via the `pepe()` system call, which returns two file descriptors: `pipefd[0]` for reading and `pipefd[1]` for writing.  After the pipe is created, the parent process can use `pipefd[1]` to write data to the pipe, and the child process can use `pipefd[0]` to read data from the pipe.  Note, this pipe has a limited buffer size (typically a few KB).

### Pipe setup and forking
We let each process `P_i` creates a pipe to `P_{i+1}`. `P_0` calls `create_children`.

```c
void create_children(int n, int pipes[][2]) {
    for (int i = 0; i < n - 1; i++) {
        pipe(pipes[i]); // Create a pipe
        pid_t pid = fork(); // Create a child process
        if (pid == 0) { // Child process
            close(pipes[i][1]); // Close the write end of the pipe
            if (i == n - 2) break; // Last child breaks the loop
        } else { // Parent process
            close(pipes[i][0]); // Close the read end of the pipe
            break;
        }
    }
}
```

In the code above, `pipes[][2]` is an array of pipes, where each pipe is an array of two file descriptors.  The first file descriptor is for reading, and the second is for writing.  Each iteration, a new pipe is created and it forks a child.  Then the child closes the write end since it only needs to read.  And the parent closes the read end since it only needs to write.  The last child breaks the loop.

In command line (shell), the `|` operator creates anonymous pipes between commands.  When we chain a series of pipes, we use the output from each process as the input for the next process.  For example, `ls -l | grep "txt"` will list all files and directories in the current directory, and then filter the output to only show files and directories that end with "txt".  A single pipe only warrants a unidirectional communication.  Two pipes are able to communication bidirectionally.  Pipes have no names (thus it's call anonymous).  Pipe is one of the most foundamental aspect of UNIX operating system.  It is an exetremely powerful notion in building process-based  parallel and distributed applications.

### Messaging

Now let's take a look at how we might encoding/decoding messages, as well as `send_message` and `receive_message`.

First we define Message as such:
```c
struct {
    int dest_id;     // Destination process ID
    int src_id;      // Source process ID
    char msg[50];    // The actual message content (up to 50 chars)
} packet;
```
Where processes forward messages based on `dest_id`.

#### Message Sending
Thus to send message, we will perform the following:
- Creates a packet with destination ID, source ID, and the message
- Safely copies the message into the packet
- Determines the correct pipe to use based on the process IDs
- Writes the entire packet to the pipe

```c
void send_message(int process, char *msg, int msg_len, int src_id, int pipes[][2], int n) {
    // Initialize the packet with destination, source, and message
    struct { int dest_id; int src_id; char msg[50]; } packet = {process, src_id, {0}};

    // Copy the message into the packet (safely with strncpy)
    strncpy(packet.msg, msg, 50);
    if (src_id < process && src_id < n - 1)
        write(pipes[src_id][1], &packet, sizeof(packet));
    else if (src_id > process)
        write(pipes[src_id - 1][1], &packet, sizeof(packet));
}
```

#### Message Receiving
To receive message, we will perform the following:
- Tries to read from all possible pipes this process might be connected to
- For each pipe:
    - If the message is for this process (packet.dest_id == curr_id), copies it to the output buffer
    - If the message is for another process, forwards it in the correct direction
    - Uses the process IDs to determine the direction to forward messages

```c
void receive_message(char *buffer, int *nread, int buffer_max, int curr_id, int pipes[][2], int n) {
    struct { int dest_id; int src_id; char msg[50]; } packet;

    // Check all possible pipes this process might read from
    for (int i = 0; i < n - 1; i++) {

        // Check if this process should read from pipe[i]
        if (curr_id == i + 1) {

            // If this message is for me, copy it to the buffer
            *nread = read(pipes[i][0], &packet, sizeof(packet));
            if (*nread > 0) {
                if (packet.dest_id == curr_id) {
                    strncpy(buffer, packet.msg, buffer_max);
                    *nread = strlen(packet.msg);
                } 
                // Otherwise, forward it to the next process if needed
                else if (packet.dest_id > curr_id && curr_id < n - 1)
                    write(pipes[curr_id][1], &packet, sizeof(packet));
            }
        }

         // Check if this process should read from pipe[i-1] (for messages going left)
        if (curr_id == i && i < n - 1) {
            *nread = read(pipes[i][0], &packet, sizeof(packet));
            if (*nread > 0 && packet.dest_id < curr_id)
                write(pipes[i - 1][1], &packet, sizeof(packet));
        }
    }
}
```

#### Message Routing
To route a message we will:
- Let each message include dest_id and src_id for routing
- Use processes to forward messages based on the destination ID
- Bidirectional Communication:
    - The system allows messages to flow in both directions
    - Each process can be both a sender and receiver
- Process Chain:
    - Processes are arranged in a linear chain
    - Each process is connected to its neighbors via pipes
- Flow Control:
    - The protocol ensures messages are only forwarded in the direction of the destination
    - Prevents infinite loops by using process IDs to determine direction

**Example Flow**
- Process 1 wants to send "Hello" to Process 3
- Process 1 calls send_message(3, "Hello", 5, 1, pipes, n)
- The message is written to the pipe between Process 1 and Process 2
- Process 2 receives the message, sees dest_id = 3 (not for itself)
- Process 2 forwards the message to Process 3
- Process 3 receives the message, sees dest_id = 3 (for itself), and processes it

#### Resource Cleanup
Finally we must `free_resources` because closing pipes is important to prevent resource leaks and ensure proper cleanup of resources.

```c
void free_resources(int curr_id, int pipes[][2], int n) {
    // Loop through all possible pipes in the system
    // There are n-1 pipes for n processes (since they're in a chain)
    for (int i = 0; i < n - 1; i++) {
        // If this process is the left end of pipe[i] (process i)
        if (curr_id == i) 
        // Close the write end of pipe[i] (index 1 is write end)
            close(pipes[i][1]);
        
         // If this process is the right end of pipe[i] (process i+1)
        if (curr_id == i + 1) 
            // Close the read end of pipe[i] (index 0 is read end)
            close(pipes[i][0]);
    }
}
```

Where:
- `curr_id`: The ID of the current process (0 to n-1)
- `pipes[][2]: 2D array where each row represents a pipe with [read_fd, write_fd]
- `n`: Total number of processes in the system


To prevent cycles we need to ensure that processes forward messages only if `dest_id != curr_id`.  To ensure convergence we use a linear chain ensures message reaches `dest_id` exactly once via directional forwarding.