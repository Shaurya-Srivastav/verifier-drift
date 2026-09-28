# Security

Report vulnerabilities to Shaurya Srivastav <ssrivastav@ucdavis.edu>. Please include a description, the version or commit, and a minimal reproduction. Do not open a public issue for an unfixed sandbox escape or for any report that includes credentials.

## Supported versions

| Version | Status |
| --- | --- |
| 0.1.0 | Unreleased. Reports against the current `main` branch are accepted. |

## Executing model-written code

MBPP grading runs untrusted Python. `vfates.tasks.sandbox.run_python` starts `sys.executable -I` in a fresh temporary directory, with a minimal environment and `stdin` closed. In `preexec_fn` it applies CPU-time, file-size and process-count limits. A wall-clock timeout kills the process group and any descendants that have left it. MBPP runs the base tests and the MBPP+ tests in separate sandboxes, and caps each program's printed output at 262,144 bytes per stream; overflow fails that stage.

That subprocess is not a security boundary against a determined author.

- Isolated mode drops user site-packages and `PYTHONPATH`. The child can still import packages installed in the same virtualenv.
- Relative writes stay in the temporary directory. Absolute paths and `..` are not jailed, so a write to `/tmp` succeeds. There is no filesystem jail.
- `RLIMIT_NPROC` is a per-user cap, not a budget for the sandboxed tree. The sandbox sets 64 when the user is already under that count, and otherwise the live process count plus 32. That stops a fork bomb, but it does not confine the sandbox alone.
- A child can leave the process group with `setsid`. The timeout records descendant pids before the process-group kill and sends `SIGKILL` to each. A grandchild that is reparented before that snapshot can still escape.
- `RLIMIT_AS` is not enforced on macOS (`setrlimit` returns `EINVAL`). The memory argument is applied only where the OS supports it, which excludes Darwin, so one large allocation can still stall the host before the timeout.
- `run_python` requires a POSIX system.

Run any training or grading job that executes model-written code inside a container or a VM. Do not point the sandbox at a host environment that holds credentials, datasets you cannot republish, or anything else you would not hand to an untrusted process.

## Secrets

Do not commit API keys, Hugging Face tokens, weights, or the private chat exports matched by `cursor_*.md`. The environment report (`vfates.cli env`) records package versions and the git HEAD. It does not record secrets, and it should stay that way.
