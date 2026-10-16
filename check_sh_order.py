#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""检查 shell 脚本里 "函数在定义之前被调用" —— bash -n 查不出这类错误。

真实踩过的坑：在 start.sh 里把用到 warn() 的代码插到了 warn() 定义【之前】，
容器里报 `start.sh: line 89: warn: command not found`，脚本第 4 行就死，
而 `bash -n` 语法检查完全通过（语法没错，是语义错）。

用法：python _check_sh_order.py <脚本...>
"""
import re
import sys
from pathlib import Path

BUILTINS = {
    "set", "export", "unset", "shift", "source", "cd", "pwd", "echo", "printf",
    "read", "exit", "return", "local", "declare", "eval", "exec", "trap", "wait",
    "sleep", "test", "true", "false", "let", "type", "command", "hash", "ulimit",
    "umask", "getopts", "kill", "jobs", "fg", "bg", "alias", "unalias", "time",
    "if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done",
    "case", "esac", "in", "function", "break", "continue", "mapfile", "dirname",
    "basename", "realpath", "readlink", "seq", "expr", "bc", "sync", "touch",
}
EXTERNALS = {
    "bash", "sh", "python", "python3", "pip", "uv", "git", "nvidia-smi", "awk",
    "sed", "grep", "find", "ls", "cat", "tee", "head", "tail", "wc", "tr", "mkdir",
    "rm", "cp", "mv", "chmod", "df", "du", "sort", "uniq", "xargs", "nohup", "env",
    "date", "hostname", "nproc", "which", "curl", "wget", "tar", "docker", "make",
    "gcc", "ldconfig", "apt-get", "apt", "sudo", "comm", "stat", "file", "timeout",
    "ssh", "scp", "rsync", "yes", "dd", "tqdm", "sleep", "true", "false",
}

FUNC_DEF = re.compile(r"^\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{")
CALL = r"(?:^|[\s;|&(]){fn}\s*(?=[\s\)]|$|;|\||&|>|<)"


def strip_heredocs(lines):
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = re.search(r"<<-?\s*'?([A-Za-z_][A-Za-z0-9_]*)'?", line)
        if m:
            tag = m.group(1)
            i += 1
            while i < len(lines):
                if lines[i].strip() == tag:
                    out.append(lines[i])
                    break
                out.append("")
                i += 1
        i += 1
    return out


def check(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = strip_heredocs(raw)
    code = [re.sub(r"^\s*#.*$", "", ln) for ln in lines]

    defs = {}
    for idx, ln in enumerate(code, 1):
        m = FUNC_DEF.match(ln)
        if m:
            defs.setdefault(m.group(1), idx)

    # 花括号嵌套深度：只有【顶层】(depth==0) 的调用才立即执行。
    # 函数体内的调用是延迟执行的 —— 区分它们可避免误报：
    #   真 bug（start.sh 曾犯）：顶层 `warn "..."` 早于 warn() 定义 -> 第 4 行就炸
    #   误报（setup_env.sh）  ：run_sync() 体内的 ok_sync()，而 run_sync 在
    #                           ok_sync 定义之后才被调用
    depths, d = [], 0
    for ln in code:
        depths.append(d)
        d = max(0, d + ln.count("{") - ln.count("}"))

    problems = []
    deferred = []
    for fn, defline in sorted(defs.items()):
        pat = re.compile(CALL.format(fn=re.escape(fn)))
        for idx in range(1, defline):
            if pat.search(code[idx - 1]):
                msg = (f"  [X] '{fn}' called at L{idx} but defined at L{defline}\n"
                       f"      L{idx}: {raw[idx-1].strip()[:110]}")
                if depths[idx - 1] == 0:
                    problems.append(msg)       # 顶层：必然报 command not found
                else:
                    deferred.append(msg)       # 函数体内：延迟执行，通常安全
                break

    known = set(defs) | BUILTINS | EXTERNALS
    unknown = {}
    for idx, ln in enumerate(code, 1):
        if FUNC_DEF.match(ln) or not ln.strip():
            continue
        if re.match(r"^\s*(export\s+)?[A-Za-z_][A-Za-z0-9_]*=", ln):
            continue
        for m in re.finditer(r"(?:^|[\s;|&(])([A-Za-z_][A-Za-z0-9_]*)\s*(?=[\s\)]|$|;|\||&|>|<)", ln):
            name = m.group(1)
            if name in known:
                continue
            before = ln[: m.start(1)].rstrip()
            if before.endswith(("$", "{", "=", "/", ".", "-", "'", '"')):
                continue
            unknown.setdefault(name, []).append(idx)

    print(f"文件: {path.name}")
    print(f"  函数定义 {len(defs)} 个: {sorted(defs)}")
    if problems:
        print(f"  !!! {len(problems)} 处【顶层】先调用后定义 —— 必然报 command not found !!!")
        for p in problems:
            print(p)
    else:
        print("  [OK] 没有顶层'先调用后定义'")
    if deferred:
        print(f"  [i] {len(deferred)} 处出现在函数体内（延迟执行，通常安全，建议人工确认）:")
        for p in deferred:
            print(p)
    susp = {k: v for k, v in unknown.items() if len(k) > 2}
    if susp:
        print(f"  [?] {len(susp)} 个可能是拼错/未定义的调用（正则误报居多，忽略即可）")
        for k, v in sorted(susp.items()):
            print(f"      {k}  (L{v[0]}...)")
    print()
    return 1 if problems else 0


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rc = 0
    for a in sys.argv[1:]:
        p = Path(a)
        if p.is_file():
            rc |= check(p)
    return rc


if __name__ == "__main__":
    sys.exit(main())
