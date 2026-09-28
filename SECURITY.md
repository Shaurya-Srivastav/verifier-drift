# Security

Report vulnerabilities to Shaurya Srivastav <ssrivastav@ucdavis.edu>. Please include a description, the version or commit, and a minimal reproduction. Do not open a public issue for an unfixed sandbox escape or for any report that includes credentials.

## Supported versions

| Version | Status |
| --- | --- |
| 0.1.0 | Unreleased. Reports against the current `main` branch are accepted. |

## Executing model-written code

Code grading (`python -I code_grader.py --once`) is the path that executes model-written Python. It does not run the submission in the process that holds the tests. Before that process reads the job, it forks a zygote. The zygote never sees expected outputs. For each test it forks a worker and sends only paths: the solution file, that test's stdin or arguments file, and the resource limits. The worker runs the submission and writes stdout, or a JSON return value, to a file. `--once` reads that file and compares it with the expected output. Expected outputs are not written to disk. A test directory's parent holds only that test's files, and only while the test runs. The solution file sits outside that directory. The process that executes submitted code therefore has this test's input and not the expected output, and not any other test's input.

After the fork, and before the submission runs, the worker adds two filesystem checks. `sys.dont_write_bytecode` is set first.

- Audit hook, every platform. `sys.addaudithook` is installed last and cannot be removed. It raises `PermissionError` for `open` reads outside the interpreter stdlib and site-packages, the solution directory and the test slot, and for writes outside the slot. That includes `/proc` and `/dev`, except `/dev/null` and `/dev/urandom`. It also denies `os.listdir`, `os.scandir` and `glob.glob` outside the slot, `os.chdir` and `os.chroot`, imports of `ctypes`, `_ctypes`, `cffi`, `_posixsubprocess`, `multiprocessing`, `_interpreters`, `_xxsubinterpreters` and `mmap`, `subprocess.Popen`, `os.system`, `os.exec`, `os.posix_spawn`, `os.spawn`, `os.fork`, `os.forkpty` and `pty.spawn`, `os.kill` and `os.killpg` aimed at any other pid, `signal.pthread_kill` aimed at any other thread, and every `socket.*` event. Ordinary stdlib imports stay allowed, as do `_thread.start_new_thread`, `threading.stack_size` and `sys.setrecursionlimit`. The worker writes the first eight denied events (event name and a short argument) into the slot. `--once` returns `n_denied` and `denied` on the grade. A denial is recorded and is not a penalty: the reward is still the tests. `n_denied` is copied onto the per-sample reward record so a later analysis can count escape attempts per arm. `fs_isolation` is `audit_hook` when this is the only filesystem check.
- Landlock, Linux only, when the kernel supports ABI 1 or newer. The worker calls `prctl(PR_SET_NO_NEW_PRIVS)` and then `landlock_create_ruleset`, `landlock_add_rule` and `landlock_restrict_self` through ctypes. The ruleset grants read and execute on the interpreter prefix trees (`sys.base_prefix` and `sys.prefix`, realpath), read on the solution directory, and read and write on the slot, plus `/dev/null` and `/dev/urandom`. It does not grant the repo root, the home directory as a whole, or the Hugging Face cache. When that restriction is installed, `fs_isolation` is `landlock+audit_hook`. macOS has no Landlock, so a Mac worker stays at `audit_hook`.
- Non-dumpable holders, Linux only. `--once` and each GraderPool server call `prctl(PR_SET_DUMPABLE, 0)` at start, before a child runs submitted code. A same-uid process then cannot open `/proc/<pid>/mem` or `/proc/<pid>/fd` for those two holders. The trainer is also an ancestor and holds the dataset. The worker's audit hook and, on Linux, Landlock already refuse `/proc/<trainer>/...`, so the trainer is not changed.

What still gets through, stated plainly:

- A native extension that is already in site-packages can issue raw syscalls. The audit hook does not see those calls. Landlock still applies to them when the kernel installed it.
- A kernel without Landlock, such as RHEL 8's 4.18, leaves only the audit hook on the filesystem.
- Run any training or grading job that executes model-written code inside a container or a VM. Do not point the sandbox at a host environment that holds credentials, datasets you cannot republish, or anything else you would not hand to an untrusted process.

## Secrets

Do not commit API keys, Hugging Face tokens, weights, or the private chat exports matched by `cursor_*.md`. The environment report (`vfates.cli env`) records package versions and the git HEAD. It does not record secrets, and it should stay that way.
