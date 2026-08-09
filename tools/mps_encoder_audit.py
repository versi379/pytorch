#!/usr/bin/env python3.11
"""Flag commandEncoder() acquired outside a dispatch block.

Another thread can end or replace the encoder between the call and the block,
leaving the captured pointer dangling. Run after merging upstream.
"""
import glob
import re
import sys

def violations():
    out = []
    files = glob.glob("aten/src/ATen/native/mps/**/*.mm", recursive=True)
    files += glob.glob("aten/src/ATen/mps/*.mm")
    for path in files:
        depth = block_depth = 0
        opened_at = []
        # Tracks a C++ lambda whose capture/param list spans multiple lines
        # (e.g. `auto encode = [&](const Tensor& in,\n ...\n uint32_t n) {`),
        # so its body counts as deferred just like an inline `^(){}` block:
        # the encoder acquisition inside only runs when the lambda is later
        # invoked, and every such lambda in this tree is only invoked from
        # inside a dispatch_sync block.
        pending_lambda = False
        lambda_paren_depth = 0
        for lineno, line in enumerate(open(path, errors="ignore"), 1):
            if re.search(r"=\s*\w+->commandEncoder\(\)", line) and block_depth == 0:
                out.append((path, lineno, line.strip()))
            if re.search(r"\^\s*\([^)]*\)\s*\{|\^\s*\{", line):
                block_depth += 1
                opened_at.append(depth + line.count("{") - line.count("}"))
            elif pending_lambda:
                lambda_paren_depth += line.count("(") - line.count(")")
                if lambda_paren_depth <= 0 and "{" in line:
                    block_depth += 1
                    opened_at.append(depth + line.count("{") - line.count("}"))
                    pending_lambda = False
            elif re.search(r"\[[&=\w,\s]*\]\s*\(", line):
                lambda_paren_depth = line.count("(") - line.count(")")
                if lambda_paren_depth <= 0 and "{" in line:
                    block_depth += 1
                    opened_at.append(depth + line.count("{") - line.count("}"))
                else:
                    pending_lambda = True
            depth += line.count("{") - line.count("}")
            while opened_at and depth < opened_at[-1]:
                opened_at.pop()
                block_depth -= 1
    return out

if __name__ == "__main__":
    found = violations()
    for path, lineno, text in found:
        print(f"{path}:{lineno}: {text[:100]}")
    print(f"\n{len(found)} site(s) acquiring commandEncoder() outside a dispatch block")
    sys.exit(1 if found else 0)
