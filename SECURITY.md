# Security

Report vulnerabilities to Shaurya Srivastav <ssrivastav@ucdavis.edu>. Please include a description, the version or commit, and a minimal reproduction. Do not open a public issue for an unfixed sandbox escape or for any report that includes credentials.

## Supported versions

| Version | Status |
| --- | --- |
| 0.1.0 | Unreleased. Reports against the current `main` branch are accepted. |

## Executing model-written code

MBPP grading runs untrusted Python. `vfates.tasks.sandbox.run_python` starts `sys.executable -I` in a fresh temporary directory, with a minimal environment, CPU-time and file-size limits applied in `preexec_fn`, and a wall-clock timeout that kills the process group.

That subprocess is not a security boundary against a determined author.

- Isolated mode drops user site-packages and `PYTHONPATH`. The child can still import packages installed in the same virtualenv.
- Relative writes stay in the temporary directory. Absolute paths and `..` are not jailed.
- The child can leave the process group with `setsid`, after which the timeout's process-group kill does not reach it.
- `RLIMIT_AS` is not enforced on macOS (`setrlimit` returns `EINVAL`). The memory argument is applied only where the OS supports it, which excludes Darwin.
- `run_python` requires a POSIX system.

Run any training or grading job that executes model-written code inside a container or a VM. Do not point the sandbox at a host environment that holds credentials, datasets you cannot republish, or anything else you would not hand to an untrusted process.

## Secrets

Do not commit API keys, Hugging Face tokens, weights, or the private chat exports matched by `cursor_*.md`. The environment report (`vfates.cli env`) records package versions and the git HEAD. It does not record secrets, and it should stay that way.
